# -*- coding: utf-8 -*-


class BaseCondition:
    """Base condition class.

    Attributes:
        hyperparam: hyperparameter affected by this condition.
        references: List of hyperparameters that this condition refers to.
        true: value to assign to the hyperparameter when the condition is met.
        false: value to assign to the hyperparameter when the condition is not met.
    """

    def __init__(self, hyperparam, true, false):
        self._hyperparam = hyperparam
        self._true = true
        self._false = false

    def evaluate(self, values):
        """Return the value or hyperparameter that needs to be used.

        The condition is evaluated using the reference hyperparameter values as input
        and either a value or a hyperparameter instance is returned.

        Args:
            values (dict): dict containing the values of the reference hyperparameters.

        Returns:
            Union[object, BaseHyperParam]: value to use
        """
        pass
