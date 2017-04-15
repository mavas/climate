"""
Return the rank of a given set of documents in ascending order.
"""


import math


corpus = ('the', 'cat', 'dog')


def dot_product(a, b):
    """Computes the dot product between 2 vectors."""
    assert len(a) == len(b)
    return sum([a*b for (a, b) in zip(a, b)])


def magnitude(a):
    """Computes the magnitude of a vector."""
    return math.sqrt(dot_product(a, a))


def _compute_vector_for_document(d):
    """Computes the vector for a document."""
    return [1 if t in d else 0 for t in corpus]


def _compute_vector_for_query(q):
    """Computes the vector for a query."""
    return [1 if q == t else 0 for t in corpus]


def _top(d, q):
    """Computes the top portion of the scoring algorithm."""
    return dot_product(d, q)


def _bottom(d, q):
    """Computes the bottom portion of the scoring algorithm."""
    return magnitude(d) * magnitude(q)


def score_document(q, d):
    """
    Computes a score of the given document with respect to the given query.

    The algorithm used for scoring is:

       d dot q
     -----------
    ||d|| x ||q||

    That is, the top portion is the dot product between the document vector and
    the query vector.  The bottom portion is the magnitude of the first vector
    multiplied by the magnitude of the second.
    """

    q = _compute_vector_for_query(q)
    d = _compute_vector_for_document(d)

    r = _top(d, q)/_bottom(d, q)

    return r


def rank_documents(q, docs):
    """Returns the rank of the given documents in ascending order."""
    r = []
    for d in docs:
        r.append(score_document(q, d))
    return sorted(r)


if __name__ == '__main__':
    assert score_document('dog', ["the", "dog"]) == 1/math.sqrt(2)
