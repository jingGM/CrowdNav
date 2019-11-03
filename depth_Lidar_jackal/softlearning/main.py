import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
import ray
from ray import tune
from ray.autoscaler.commands import exec_cluster

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import (get_policy_from_variant, get_policy_from_params)
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables,datetimestamp, PROJECT_PATH

tf.compat.v1.disable_eager_execution()


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self._session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(self._session)

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        variant = copy.deepcopy(self._variant)

        environment_params = variant['environment_params']

        training_environment = self.training_environment = (get_environment_from_params(environment_params['training']))

        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(variant, training_environment)
        policy = self.policy = get_policy_from_variant(variant, training_environment)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(variant['exploration_policy_params'], training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    @property
    def picklables(self):
        return {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'policy_weights': self.policy.get_weights(),
        }

    def _save_value_functions(self, checkpoint_dir):
        if isinstance(self.Qs, tf.keras.Model):
            Qs = [self.Qs]
        elif isinstance(self.Qs, (list, tuple)):
            Qs = self.Qs
        else:
            raise TypeError(self.Qs)

        for i, Q in enumerate(Qs):
            checkpoint_path = os.path.join(checkpoint_dir,f'Qs_{i}')
            Q.save_weights(checkpoint_path)

    def _restore_value_functions(self, checkpoint_dir):
        if isinstance(self.Qs, tf.keras.Model):
            Qs = [self.Qs]
        elif isinstance(self.Qs, (list, tuple)):
            Qs = self.Qs
        else:
            raise TypeError(self.Qs)

        for i, Q in enumerate(Qs):
            checkpoint_path = os.path.join(checkpoint_dir,f'Qs_{i}')
            Q.load_weights(checkpoint_path)

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.picklables, f)

        self._save_value_functions(checkpoint_dir)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        tf_checkpoint = self._get_tf_checkpoint()

        tf_checkpoint.save(file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        training_environment = self.training_environment = picklable['training_environment']
        evaluation_environment = self.evaluation_environment = picklable['evaluation_environment']

        replay_pool = self.replay_pool = (get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = get_Q_function_from_variant(self._variant, training_environment)
        self._restore_value_functions(checkpoint_dir)
        policy = self.policy = (get_policy_from_variant(self._variant, training_environment))
        self.policy.set_weights(picklable['policy_weights'])

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(self._variant['exploration_policy_params'],training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu

    return resources

def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec

def generate_experiment_kwargs(variant_spec, command_line_args):
    # TODO(hartikainen): Allow local dir to be modified through cli args
    local_dir = os.path.join(
        '~/ray_results',
        command_line_args.universe,
        command_line_args.domain,
        command_line_args.task)
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, command_line_args.exp_name))

    variant_spec = add_command_line_args_to_variant_spec(variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None

        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)

        return tune.function(trial_name_creator)

    experiment_kwargs = {
        'name':                experiment_id,
        'resources_per_trial': resources_per_trial,
        'config':              variant_spec,
        'local_dir':           local_dir,
        'num_samples':         command_line_args.num_samples,
        'upload_dir':          command_line_args.upload_dir,
        'checkpoint_freq':     (variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end':   (variant_spec['run_params']['checkpoint_at_end']),
        'max_failures':        command_line_args.max_failures,
        'trial_name_creator':  create_trial_name_creator(command_line_args.trial_name_template),
        'restore':             command_line_args.restore,  # Defaults to None
    }

    return experiment_kwargs

def get_trainable_class(*args, **kwargs):
    return ExperimentRunner


def get_variant_spec(command_line_args, *args, **kwargs):
    from variants import get_variant_spec
    variant_spec = get_variant_spec(command_line_args, *args, **kwargs)
    return variant_spec


def get_parser():
    from utils import get_parser
    parser = get_parser()
    return parser


def run_example_local(example_argv, local_mode=False):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    example_args = get_parser().parse_args(example_argv)
    variant_spec = get_variant_spec(example_args)
    trainable_class = get_trainable_class(example_args)

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

    ray.init(
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=example_args.resources or {},
        local_mode=local_mode,
        include_webui=example_args.include_webui,
        temp_dir=example_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        with_server=example_args.with_server,
        server_port=example_args.server_port,
        scheduler=None,
        reuse_actors=True)


def main(argv=None):
    run_example_local(argv)

if __name__ == '__main__':
    main(argv=sys.argv[1:])

