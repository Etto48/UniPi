from __future__ import annotations
from typing import OrderedDict, override

class BAN:
    def __init__(self, values: list[tuple[int, float]]):
        values.sort(key=lambda x: x[0], reverse=True)
        # check if duplicate degrees
        for i in range(len(values) - 1):
            if values[i][0] == values[i + 1][0]:
                raise ValueError(f"Duplicate degree found: {values[i][0]}")
        self.values = OrderedDict(values)

    def copy(self) -> BAN:
        return BAN(list(self.values.items()))

    @staticmethod
    def alpha() -> BAN:
        return BAN([(1, 1.0)])

    @staticmethod
    def eta() -> BAN:
        return BAN([(-1, 1.0)])

    @staticmethod
    def _try_into_ban(value: float | int | BAN) -> BAN:
        if isinstance(value, BAN):
            return value
        elif isinstance(value, (float, int)):
            return BAN([(0, float(value))])
        else:
            raise TypeError(f"Cannot convert {type(value)} to BAN")

    @override
    def __add__(self, other: float| int | BAN) -> BAN:
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot add {type(other)} to BAN") from e
        new_values = []
        all_degrees = set(self.values.keys()).union(other.values.keys())
        for degree in all_degrees:
            value = self.values.get(degree, 0.0) + other.values.get(degree, 0.0)
            if value != 0.0:
                new_values.append((degree, value))
        return BAN(new_values)
    
    @override
    def __sub__(self, other: float| int | BAN) -> BAN:
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot subtract {type(other)} from BAN") from e
        new_values = []
        all_degrees = set(self.values.keys()).union(other.values.keys())
        for degree in all_degrees:
            value = self.values.get(degree, 0.0) - other.values.get(degree, 0.0)
            if value != 0.0:
                new_values.append((degree, value))
        return BAN(new_values)
    
    @override
    def __mul__(self, other: float| int | BAN) -> BAN:
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot multiply {type(other)} with BAN") from e
        new_values = {}
        for degree, value in self.values.items():
            for other_degree, other_value in other.values.items():
                new_degree = degree + other_degree
                new_value = value * other_value
                if new_degree in new_values:
                    new_values[new_degree] += new_value
                else:
                    new_values[new_degree] = new_value
        return BAN(list(new_values.items()))

    @override
    def __eq__(self, other: float | int | BAN) -> bool:
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot compare {type(other)} with BAN") from e
        if len(self.values) != len(other.values):
            return False
        for (this_degree, this_value), (other_degree, other_value) in zip(self.values.items(), other.values.items()):
            if this_degree != other_degree or this_value != other_value:
                return False
        return True
    
    @override
    def __ne__(self, other: float | int | BAN) -> bool:
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot compare {type(other)} with BAN") from e
        return not self.__eq__(other)
    

    def compare(self, other: float | int | BAN) -> int:
        """
        Compare this BAN with another BAN or a numeric value.
        Returns:
            -1 if this BAN is less than the other,
             0 if they are equal,
             1 if this BAN is greater than the other.
        """
        try:
            other = self._try_into_ban(other)
        except TypeError as e:
            raise TypeError(f"Cannot compare {type(other)} with BAN") from e
        degrees = set(self.values.keys()).union(other.values.keys())
        ordered_degrees = sorted(degrees, reverse=True)
        for degree in ordered_degrees:
            self_value = self.values.get(degree, 0.0)
            other_value = other.values.get(degree, 0.0)
            if self_value < other_value:
                return -1
            elif self_value > other_value:
                return 1
        return 0
    
    @override
    def __lt__(self, other: float | int | BAN) -> bool:
        return self.compare(other) < 0
    
    @override
    def __le__(self, other: float | int | BAN) -> bool:
        return self.compare(other) <= 0
    
    @override
    def __gt__(self, other: float | int | BAN) -> bool:
        return self.compare(other) > 0
    
    @override
    def __ge__(self, other: float | int | BAN) -> bool:
        return self.compare(other) >= 0
    
    @override
    def __hash__(self) -> int:
        return hash(tuple(self.values.items()))
    
    @override
    def __repr__(self) -> str:
        return f"BAN({list(self.values.items())})"
    
    @override
    def __str__(self) -> str:
        ret = ""
        alpha = "α"
        eta = "η"
        first = True
        for degree, value in self.values.items():
            
            if value >= 0.0 and not first:
                sign= "+"
            elif value >= 0.0 and first:
                sign = ""
            else:
                sign = "-"
            if first:
                first = False
            if degree == 0:
                mult = ""
            elif degree == 1:
                mult = alpha
            elif degree == -1:
                mult = eta
            elif degree > 0:
                mult = f"{alpha}^{degree}"
            elif degree < 0:
                mult = f"{eta}^{-degree}"

            ret += f"{sign} {abs(value):.2f}{mult} "
        return ret.strip()
    

if __name__ == "__main__":
    ban1 = BAN([(1, 2.0), (0, 3.0)])
    ban2 = BAN([(1, 1.0), (-1, 4.0)])
    print(ban1 + ban2)  # Should print a new BAN with combined values
    print(ban1 - ban2)  # Should print a new BAN with subtracted values
    print(ban1 * ban2 * ban1 * ban2)  # Should print a new BAN with multiplied values
    print(ban1 == ban2)  # Should print False
    print(ban1 != ban2)  # Should print True
    print(ban1.compare(ban2))  # Should print -1, 0, or 1 based on comparison
    print(ban1 < ban2)  # Should print True/False based on comparison
    print(ban1 <= ban2)  # Should print True/False based on comparison
    print(ban1 > ban2)  # Should print True/False based on comparison
    print(ban1 >= ban2)  # Should print True/False based on comparison
    print(ban1)  # Should print the string representation of ban1