# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BaseHyperParam(metaclass=ABCMeta):
    """Abstract representation of a single hyperparameter that needs to be tuned.

    Attributes:
        name (hashable): Name of this HyperParam.
        _k (int): Number of dimensions that this HyperParam uses to be represented in
                  the search space.
    """

    @abstractmethod
    def transform(self, values):
        """Transform one or more hyperparameter values.

        Transform one or more hyperparameter values from the original hyperparameter space to the
        normalized search space [0, 1]^k.

        Args:
            values (Union[object, List[object]]): single value or list of values to normalize.

        Returns:
            normalized (ArrayLike): 2D array of shape(len(values), self._k)
        """
        pass

    @abstractmethod
    def reverse_transform(self, values):
        """Revert one or more hyperparameter values.

        Transform one or more hyperparameter values from the normalized search
        space [0, 1]^k to the original hyperparameter space.

        Args:
            values (ArrayLike): single value or 2D ArrayLike of normalized values.

        Returns:
            denormalized (Union[object, List[object]]): denormalized value or list of denormalized
                                                        values.
        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """Sample values in this hyperparameter search space.
        Args:
            n_samples (int): Number of values to sample.

        Returns:
            samples (ArrayLike): 2D array with of shape (n_samples, self._k)
        """
        pass

    @abstractmethod
    def to_dict(self):
        """Get a dict representation of this HyperParam."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, spec_dict):
        """Load a HyperParam from a dcit representation."""
        pass
