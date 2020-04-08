import LinearAlgebra

@inline function _get_diagonal_entries(m::AbstractMatrix)
    LinearAlgebra.checksquare(m)
    return LinearAlgebra.diag(m)
end

@inline function _pseudoinverse(m::AbstractMatrix)
    try
        return LinearAlgebra.inv(m)
    catch
    end
    return LinearAlgebra.pinv(m)
end
