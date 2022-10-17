from typing import Dict, List, Callable, Union
import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, List[float]],
                         scoring: Callable[[Union[Dataset, np.array], np.array], float] = None,
                         cv: int = 3,
                         n_iter: int = 1000,
                         test_size: float = 0.2) -> Dict[str, List[float]]:
  """
  Performs a random search on the model with the given parameter distribution and cross-validates the model using the
  given dataset, scoring function, and number of folds.

  :param model: model to cross validate
  :param dataset: a given dataset
  :param parameter_distribution: a dictionary with the parameter names and the corresponding distribution of values
  :param scoring: score function
  :param cv: number of folds
  :param test_size: the size of the test set
  """
  # Initialize the results dictionary
  scores = {
    'parameters': [],
    'seeds': [],
    'train': [],
    'test': []
  }

  # Check if the given parameters exist in the model
  for parameter in parameter_distribution:
    if not hasattr(model, parameter):
      raise ValueError(f"{parameter} is not a valid parameter for the model.")

  # Get n_iter combinations of parameters
  for combination in range(n_iter):
    # Generate a random seed
    random_state = np.random.randint(0, 1000)

    # Save the seed
    scores['seeds'].append(random_state)

    # Initialize the parameters dictionary
    parameters = {}

    # Set the parameters for the model
    for parameter, value in parameter_distribution.items():
      # Select a random value from the distribution of values for each parameter
      parameters[parameter] = np.random.choice(value)

    # Set the parameters to the model
    for parameter, value in parameters.items():
      setattr(model, parameter, value)

    # Cross-validate the model with the current parameter combination
    score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

    # Save the parameter combination and the obtained scores
    scores['parameters'].append(parameters)
    scores['train'].append(score['train'])
    scores['test'].append(score['test'])

  # Return the results
  return scores

