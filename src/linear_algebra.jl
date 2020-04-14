import LinearAlgebra

function _pseudoinverse(m::AbstractMatrix)
    try
        return LinearAlgebra.inv(m)
    catch ex
        @debug("", exception = (ex, catch_backtrace()))
    end
    return LinearAlgebra.pinv(m)
end
