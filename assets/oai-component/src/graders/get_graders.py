from graders.equality_garder import EqualityGrader


def get_grader(identifier):
    if identifier == "Equality":
        return EqualityGrader
    raise ValueError(f"Unknown identifier: {identifier}")