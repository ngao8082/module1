from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    x = vals[arg]

    # Use the central difference formula to approximate the derivative
    f_plus = f(*vals[:arg], x + epsilon, *vals[arg+1:])
    f_minus = f(*vals[:arg], x - epsilon, *vals[arg+1:])
    return (f_plus - f_minus) / (2 * epsilon)
    raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
     
    
    visited = set()
    queue = [variable]
    while queue: 
        var = queue.pop(0)
        if var not in visited:
            visited.add(var)
            if not var.constant:  
                yield var
                queue.extend(var.inputs)
    if len(visited) != len(variable.graph.variables):
        raise Exception('Computation graph is not connected')
    raise NotImplementedError('Need to implement for Task 1.4')


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    
    variable.accumulate_derivative(deriv)
    derivatives = {variable: deriv}
    for var in reversed(list(topological_sort(variable))):
        deriv = var.backward(derivatives[var])
        for i, input_var in enumerate(var.inputs):
            input_deriv = deriv[i]
            input_var.accumulate_derivative(input_deriv)
            if input_var in derivatives:
                derivatives[input_var] += input_deriv
            else:
                derivatives[input_var] = input_deriv
    raise NotImplementedError('Need to implement for Task 1.4')


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
