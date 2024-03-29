# -*- coding: utf-8 -*-
'''
LightGBM callback

動的な学習率管理用スクリプト
'''



def strategy_func(current_lr, eval_history, best_round):
    """
    :param current_lr: 現在の学習率 (指定されていない場合の初期値は None)
    :param eval_history: 検証用データに対する評価指標の履歴
    :param best_round: 現状で最も評価指標の良かったラウンド数
    :return: 次のラウンドで用いる学習率
    """
    current_lr = current_lr or 0.1    
    if (len(eval_history) - best_round == 10):
#       print('set new learning rate {} .>.>. {}'.format(current_lr,current_lr/2))
      current_lr /=2
    min_threshold = 0.001
    current_lr = max(min_threshold, current_lr)

    return current_lr

class DynamicallyLrControllerCallback(object):

    def __init__(self, strategy_func):
        # 学習率を決定するための関数
        self.strategy_func = strategy_func
        # 検証用データに対する評価指標の履歴
        self.metric_history = []

    def __call__(self, env):
        # 現在の学習率を取得する
        current_lr = env.params.get('learning_rate')

        # 検証用データに対する評価結果を取り出す (先頭の評価指標)
        first_eval_result = env.evaluation_result_list[0]
        # スコア
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているか否か
        is_higher_better = first_eval_result[3]

        # 評価指標の履歴を更新する
        self.metric_history.append(metric_score)
        # 現状で最も優れたラウンド数を計算する
        best_round_find_func = np.argmax if is_higher_better else np.argmin
        best_round = best_round_find_func(self.metric_history)

        # 新しい学習率を計算する
        new_lr = self.strategy_func(current_lr=current_lr,
                                    eval_history=self.metric_history,
                                    best_round=best_round)

        # 次のラウンドで使う学習率を更新する
        update_params = {
            'learning_rate': new_lr,
        }
        env.model.reset_parameter(update_params)
        env.params.update(update_params)

    @property
    def before_iteration(self):
        # コールバックは各イテレーションの後に実行する
        return False

import collections
import warnings
from operator import gt, lt


class EarlyStopException(Exception):
    """Exception of early stopping."""

    def __init__(self, best_iteration, best_score):
        """Create early stopping exception.
        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
        best_score : float
            The score of the best iteration.
        """
        super(EarlyStopException, self).__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "LightGBMCallbackEnv",
    ["model",
     "params",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])


def _format_eval_result(value, show_stdv=True):
    """Format metric string."""
    if len(value) == 4:
        return '%s\'s %s: %g' % (value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return '%s\'s %s: %g + %g' % (value[0], value[1], value[2], value[4])
        else:
            return '%s\'s %s: %g' % (value[0], value[1], value[2])
    else:
        raise ValueError("Wrong metric value")


def print_evaluation(period=1, show_stdv=True):
    """Create a callback that prints the evaluation results.
    Parameters
    ----------
    period : int, optional (default=1)
        The period to print the evaluation results.
    show_stdv : bool, optional (default=True)
        Whether to show stdv (if provided).
    Returns
    -------
    callback : function
        The callback that prints the evaluation results every ``period`` iteration(s).
    """
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            print('[%d]\t%s' % (env.iteration + 1, result))
    _callback.order = 10
    return _callback


def record_evaluation(eval_result):
    """Create a callback that records the evaluation history into ``eval_result``.
    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.
    Returns
    -------
    callback : function
        The callback that records the evaluation history into the passed dictionary.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('Eval_result should be a dictionary')
    eval_result.clear()

    def _init(env):
        for data_name, _, _, _ in env.evaluation_result_list:
            eval_result.setdefault(data_name, collections.defaultdict(list))

    def _callback(env):
        if not eval_result:
            _init(env)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            eval_result[data_name][eval_name].append(result)
    _callback.order = 20
    return _callback


def reset_parameter(**kwargs):
    """Create a callback that resets the parameter after the first iteration.
    Note
    ----
    The initial parameter will still take in-effect on first iteration.
    Parameters
    ----------
    **kwargs : value should be list or function
        List of parameters for each boosting round
        or a customized function that calculates the parameter in terms of
        current number of round (e.g. yields learning rate decay).
        If list lst, parameter = lst[current_round].
        If function func, parameter = func(current_round).
    Returns
    -------
    callback : function
        The callback that resets the parameter after the first iteration.
    """
    def _callback(env):
        new_parameters = {}
        for key, value in kwargs.items():
            if key in ['num_class', 'num_classes',
                       'boosting', 'boost', 'boosting_type',
                       'metric', 'metrics', 'metric_types']:
                raise RuntimeError("cannot reset {} during training".format(repr(key)))
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError("Length of list {} has to equal to 'num_boost_round'."
                                     .format(repr(key)))
                new_param = value[env.iteration - env.begin_iteration]
            else:
                new_param = value(env.iteration - env.begin_iteration)
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)
    _callback.before_iteration = True
    _callback.order = 10
    return _callback


def early_stopping(stopping_rounds, first_metric_only=False, verbose=True):
    """Create a callback that activates early stopping.
    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]

    def _init(env):
        enabled[0] = not any((boost_alias in env.params
                              and env.params[boost_alias] == 'dart') for boost_alias in ('boosting',
                                                                                         'boosting_type',
                                                                                         'boost'))
        if not enabled[0]:
            warnings.warn('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if verbose:
            msg = "Training until validation scores don't improve for {} rounds."
            print(msg.format(stopping_rounds))

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            elif env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    print('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if first_metric_only:  # the only first metric is used for early stopping
                break
    _callback.order = 30
    return _callback

dylr_cb = DynamicallyLrControllerCallback(strategy_func=strategy_func)
callback_l = []