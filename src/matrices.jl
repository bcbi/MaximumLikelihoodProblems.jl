import LinearAlgebra

@inline function _get_diagonal_entries(m::AbstractMatrix)
    LinearAlgebra.checksquare(m)
    return LinearAlgebra.diag(m)
end
