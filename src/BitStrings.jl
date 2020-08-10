# TODO
#	- Implment broadcasting
#
#	- Consider generalizing this to static bit arrays of any dimension.
#	- Also, consider incorporating this into the BitArray framework. This would
#		require a reworking BitArray; possibly parameterize by backing type?
#		e.g.	BitArray{N} = AbstractBitArray{N, Array{N,UInt64}}
#				SBitArray{S,N} = AbstractBitArray{N, SArray{S,N,UInt64}}
#				MBitArray{S,N} = AbstractBitArray{N, MArray{S,N,UInt64}}
#		then SBitVector{L} = SBitArray{Tuple{L}, 1}, etc.
"""
	BitStrings (module)

Dense-packed, stack-allocated bit strings.
(Basically a merging of SVector and BitVector.)
"""
module BitStrings

#using BaseExtensions: allequal
using StaticArrays
import Base: convert, promote_type, promote_rule
import Base: string, print, show, display
import Base: length, size, checkbounds, ndims, axes, getindex, setindex, setindex!, iterate
import Base: vcat
import Base: map, map!		#, bit_map!
import Base.Broadcast: broadcastable, BroadcastStyle, Broadcasted, broadcasted
import Base: count, sum
import Base: +, -, *, /
import LinearAlgebra: dot

export AbstractBitString, SBitString, MBitString, parity, hammdist, setindex

# TODO:	Support for other binary element types
# TODO:	Support multidimensional arrays?


## From BaseExtensions

const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}

function allequal(it::Iterable)
	if length(it) <= 1
		return true
	end
	v = first(it)
	for v_ in it
		v_ === v || return false
	end
	return true
end


## OK, Now the actual stuff

abstract type AbstractBitString{L,N} <: AbstractVector{Bool} end

struct SBitString{L,N} <: AbstractBitString{L,N}
	chunks::SVector{N,UInt64}
end

struct MBitString{L,N} <: AbstractBitString{L,N}
	chunks::MVector{N,UInt64}
	MBitString{L,N}(ch) where {L,N} = new{L,N}(ch)
	MBitString{L,N}(::UndefInitializer) where {L,N} = new{L,N}(MVector{N,UInt64}(undef))
end


# These determine the appropriate constructor type for operations
promote_type(::Type{<:SBitString}) = SBitString
promote_type(::Type{<:MBitString}) = MBitString
# These are needed to strip the {L,N} parameters
promote_type(::Type{T}, ::Type{T}) where {T<:SBitString{L,N}} where {L,N} = SBitString
promote_type(::Type{T}, ::Type{T}) where {T<:MBitString{L,N}} where {L,N} = MBitString
promote_rule(::Type{<:SBitString}, ::Type{<:MBitString}) = MBitString

SBitString(b::Bool) = SBitString{1,1}((UInt64(b),))
MBitString(b::Bool) = MBitString{1,1}((UInt64(b),))

convert(::Type{SBitString}, b::Bool) = SBitString(b)
convert(::Type{MBitString}, b::Bool) = MBitString(b)

# AbstractBitString(bs::AbstractBitString{L,N}) where {L,N} = AbstractBitString{L,N}(bs.chunks)
#
#
# # Construction from integer
# AbstractBitString(i::Integer) = AbstractBitString{8*sizeof(typeof(i)),1}(SVector{1}(UInt64(i)))
#


# Construction from BitVector (mre efficient than for general iterables)
function (::Type{T})(b::BitVector) where T<:AbstractBitString
	T{length(b), length(b.chunks)}(b.chunks)
end

# Construction from any iterable
function (::Type{T})(A) where T<:AbstractBitString
	chunks = compute_chunks(A)
	T{length(A), length(chunks)}(chunks)
end

convert(::Type{SBitString{L,N}}, v::SVector{N,UInt64}) where {L,N} = SBitString{L,N}(v)
convert(::Type{MBitString{L,N}}, v::MVector{N,UInt64}) where {L,N} = MBitString{L,N}(v)

convert(::Type{T}, bs::T) where {T<:AbstractBitString} = bs
convert(::Type{SBitString}, bs::AbstractBitString{L,N}) where {L,N} = SBitString{L,N}(bs.chunks)
convert(::Type{SBitString{L,N}}, bs::AbstractBitString{L,N}) where {L,N} = SBitString{L,N}(bs.chunks)
convert(::Type{MBitString}, bs::AbstractBitString{L,N}) where {L,N} = MBitString{L,N}(bs.chunks)
convert(::Type{MBitString{L,N}}, bs::AbstractBitString{L,N}) where {L,N} = MBitString{L,N}(bs.chunks)



nchunks(len::Int) = Base._div64(len + 63)
nchunks(b::AbstractBitString{L,N}) where {L,N} = N

# Compute chunks from an iterable
function compute_chunks(A)
	len = length(A)
	nch = nchunks(len)

	# empty vector
	nch == 0 && return MVector{0,UInt64}()

	chunks = MVector{nch,UInt64}(undef)
	itr = iterate(A)
	@inbounds begin
		for ich = 1:nch-1
			ch = UInt64(0)
			for j = 0:63
				ch |= (UInt64(convert(Bool, itr[1])) << j)
				itr = iterate(A, itr[2])
			end
			chunks[ich] = ch
		end
		ch = UInt64(0)
		for j = 0:Base._mod64(len-1)
			ch |= (UInt64(convert(Bool, itr[1])) << j)
			itr = iterate(A, itr[2])
		end
		chunks[nch] = ch
	end
	return chunks
end

length(b::AbstractBitString{L}) where {L} = L
size(b::AbstractBitString{L}) where {L} = (L,)
ndims(b::AbstractBitString)  = 1
ndims(::Type{T}) where {T<:AbstractBitString} = 1
axes(b::AbstractBitString{L}) where {L} = (Base.OneTo(L),)

# # Conversion to BitVector
# convert(::Type{BitVector}, bs::AbstractBitString) = BitVector(bs)
# convert(::Type{AbstractBitString}, bv::BitVector) = AbstractBitString(bv)
#
# function BitVector(bs::AbstractBitString)
# 	bv = BitVector(undef, length(bs))
# 	bv.chunks = Vector(bs.chunks)
# 	bv
# end


"""
	string(bs::AbstractBitString)

Create a string representation of a `AbstractBitString`.
Bits are shown as 0,1 in order from left to right, with an underscore every 8 bits.
"""
function string(bs::AbstractBitString)
	mv = MVector{length(bs) + max(0,(length(bs)-1))>>3, Char}(undef)
	j = 1
	for i in 1:length(bs)
		mv[j] = bs[i] + '0'
		j += 1
		if mod(i, 8) == 0 && i < length(bs)
			mv[j] = '_'
			j += 1
		end
	end
	string(mv...)
end

display(bs::AbstractBitString) = show(bs)
show(io::IO, bs::SBitString) = print(io, string(bs))
show(io::IO, bs::MBitString) = print(io, "(", string(bs), ")")
#printstyled(io, string((bs.bvec .+ '0')...); bold=true)


##  Indexing

checkbounds(b::AbstractBitString{L}, I...) where {L} = Base.checkbounds_indices(Bool, (Base.OneTo(L),), I) || Base.throw_boundserror(b, I)

@inline function get_chunk_index(i::Integer)
	i1, i2 = Base.get_chunks_id(i)
	msk = UInt64(1) << i2
	return (i1, msk)
end


# getindex - entry point
@inline function getindex(b::AbstractBitString, i)
   @boundscheck checkbounds(b, i)
	_getindex(b, i)
end

# implementations (no bounds checking)
@inline _getindex(b::AbstractBitString, i::CartesianIndex{1}) = _getindex(b, i[1])
@inline _getindex(b::AbstractBitString, i::Integer) = _getindex(b, get_chunk_index(i))

# unsafe getindex using precomputed chunk index and mask
@inline function _getindex(b::AbstractBitString, (ich,msk)::Tuple{Int64, UInt64})
	# @info "_getindex (chunk): ich = $ich, msk = $msk"
	@inbounds r = (b.chunks[ich] & msk) != 0
	return r
end

# fallback / index by an iterable
function _getindex(b::T, itr) where T<:AbstractBitString
	 v = [_getindex(b, get_chunk_index(i)) for i in itr]
	 promote_type(T)(v)
end


function setindex!(bs::MBitString, val, i)
	@boundscheck checkbounds(bs, i)
	_setindex!(bs, val, i)
end


@inline function _setindex!(bs::MBitString, val::Bool, i::Integer)
	i1, i2 = Base.get_chunks_id(i)
	msk = ~(UInt64(1) << i2)
	@inbounds bs.chunks[i1] = (bs.chunks[i1] & msk) | (val << i2)
end

function _setindex!(bs::MBitString, val, itr)
	for (iv,ib) in enumerate(itr)
		_setindex!(bs, val[iv], ib)
	end
end


# function setindex(bs::AbstractBitString, val, i)
# 	@boundscheck checkbounds(bs, i)
# 	_setindex(bs, val, i)
# end
#
#
# @inline function _setindex(bs::AbstractBitString{L,N}, val::Bool, i::Integer) where {L,N}
# 	temp = MVector{N,UInt64}(bs.chunks)
# 	i1, i2 = Base.get_chunks_id(i)
# 	msk = ~(UInt64(1) << i2)
# 	@inbounds temp[i1] = (temp[i1] & msk) | (val << i2)
# 	AbstractBitString{L,N}(SVector{N,UInt64}(temp))
# end
#
#
#
# @inline function _setindex(bs::AbstractBitString{L,N}, val::AbstractVector{Bool}, itr) where {L,N}
# 	temp = MVector{N,UInt64}(bs.chunks)
# 	for (iv,ib) in enumerate(itr)
# 		i1, i2 = Base.get_chunks_id(ib)
# 		msk = ~(UInt64(1) << i2)
# 		@inbounds temp[i1] = (temp[i1] & msk) | (val[iv] << i2)
# 	end
# 	AbstractBitString{L,N}(SVector{N,UInt64}(temp))
# end
#
#
#



##  Iteration

function iterate(bs::AbstractBitString, i::Int=0)
    i >= length(bs) && return nothing
    (bs.chunks[Base._div64(i)+1] & (UInt64(1)<<Base._mod64(i)) != 0, i+1)
end



vcat(a::AbstractBitString) = a

@inline function vcat(a::AbstractBitString, b::AbstractBitString)
	lena = length(a)
	lenb = length(b)
	nch = nchunks(lena + lenb)
	na = length(a.chunks)
	nb = length(b.chunks)
	chunks = MVector{nch, UInt64}(undef)
	@inbounds for i = 1:na
		chunks[i] = a.chunks[i]
	end
	off = Base._mod64(length(a))
	@inbounds for j = 1:nb
		chunks[na + j-1]  |= b.chunks[j] << off
		if na + j <= nch
			chunks[na + j] = b.chunks[j] >> (64-off)
		end
	end
	return promote_type(typeof(a), typeof(b)){lena+lenb, nch}(chunks)
end
#
# totallength() = 0
# totallength(a::AbstractBitString) = length(a)
# totallength(a::AbstractBitString, others::AbstractBitString...) = length(a) + totallength(others...)
#
# # Without inlining it is SLOW
# @inline function vcat(args::AbstractBitString...)
# 	totlen = totallength(args...)
# 	nch = nchunks(totlen)
# 	chunks = MVector{nch, UInt64}(undef)
# 	a = args[1]
# 	for i = 1:length(a.chunks)
# 		chunks[i] = a.chunks[i]
# 	end
# 	len = length(a)
#
# 	for a in Base.tail(args)
# 		ich = nchunks(len)
# 		off = Base._mod64(len)
# 		for j = 1:length(a.chunks)
# 			chunks[ich - 1 + j]  |= a.chunks[j] << off
# 			if ich + j <= nch
# 				chunks[ich + j] = a.chunks[j] >> (64-off)
# 			end
# 		end
# 		len += length(a)
# 	end
# 	return AbstractBitString{totlen, nch}(chunks)
# end

# Basic arithmetic operations -- result in a non-dense array
-(b::AbstractBitString) = (-).(b)
*(b::AbstractBitString, x::Number) = b .* x
*(x::Number, b::AbstractBitString) = x .* b
/(b::AbstractBitString, x::Number) = b ./ x



# Efficient versions of map and map! for boolean functions.
map(::Union{typeof(~), typeof(!)}, a::AbstractBitString) = bit_map(~, a)
map(::Union{typeof(&), typeof(*), typeof(min)}, a::AbstractBitString, b::AbstractBitString) = bit_map(&, a, b)
map(::Union{typeof(|), typeof(max)}, a::AbstractBitString, b::AbstractBitString) = bit_map(|, a, b)
map(::Union{typeof(⊻), typeof(!=)}, a::AbstractBitString, b::AbstractBitString) = bit_map(⊻, a, b)
map(::typeof(*), a::AbstractBitString, b::AbstractBitString) = bit_map(*, a, b)
map(::typeof(==), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> ~xor(x,y), a, b)
map(::typeof(^), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> x | ~y, a, b)
map(::typeof(>), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> x & ~y, a, b)
map(::typeof(>=), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> x | ~y, a, b)
map(::typeof(<), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> y & ~x, a, b)
map(::typeof(<=), a::AbstractBitString, b::AbstractBitString) = bit_map((x,y) -> y | ~x, a, b)
map(::typeof(min), args...) where T = bit_map(&, args...)
map(::typeof(max), args...) = bit_map(|, args...)


# 1-ary functions
@inline function bit_map(f::F, a::AbstractBitString{L,N}) where {L,N,F}
	temp = MVector{N,UInt64}(undef)
#	isempty(A) && return AbstractBitString{0,1}()
	for i = 1:N
   	temp[i] = f(a.chunks[i])
   end
   temp[N] &= Base._msk_end(L)
   promote_type(typeof(a))(temp)
end


# 2-ary functions
function bit_map(f::F, a::AbstractBitString{L,N},  b::AbstractBitString{L,N}) where {F,L,N}
	#	isempty(A) && return AbstractBitString{0,1}()
	temp = MVector{N,UInt64}(undef)
	for i = 1:N
   	temp[i] = f(a.chunks[i], b.chunks[i])
   end
   temp[N] &= Base._msk_end(L)
   promote_type(typeof(a), typeof(b)){L,N}(temp)
end

# n-ary functions
@inline function bit_map(f::F, args::AbstractBitString{L,N}...) where {F,L,N}
	allequal(length(arg) for arg in args) || throw(DimensionMismatch("sizes of all arguments must match"))
	temp = MVector{N,UInt64}(undef)
	for i = 1:N
   	temp[i] = f((arg.chunks[i] for arg in args)...)
   end
   temp[N] &= Base._msk_end(L)
   AbstractBitString{L,N}(SVector{N}(temp))
end
#



map!(::Union{typeof(~), typeof(!)}, dest::MBitString, a::AbstractBitString) = bit_map!(~, dest, a)
map!(::Union{typeof(&), typeof(*), typeof(min)}, a::AbstractBitString, b::AbstractBitString) = bit_map!(&, dest, a, b)
map!(::Union{typeof(|), typeof(max)}, a::AbstractBitString, b::AbstractBitString) = bit_map!(|, dest, a, b)
map!(::typeof(⊻), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!(⊻, dest, a, b)
map!(::typeof(*), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!(*, dest,a, b)
map!(::typeof(!=), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!(xor, dest, a, b)
map!(::typeof(==), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> ~xor(x,y), dest, a, b)
map!(::typeof(^), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> x | ~y, dest, a, b)
map!(::typeof(>), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> x & ~y, dest, a, b)
map!(::typeof(>=), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> x | ~y, dest, a, b)
map!(::typeof(<), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> y & ~x, dest, a, b)
map!(::typeof(<=), dest::MBitString, a::AbstractBitString, b::AbstractBitString) = bit_map!((x,y) -> y | ~x, dest, a, b)
map!(::typeof(min), dest::MBitString, args::AbstractBitString...) where T = bit_map!(&, dest, args...)
map!(::typeof(max), dest::MBitString, args::AbstractBitString...) = bit_map!(|, dest, args...)



@inline function bit_map!(f, dest, args...)
	checklengths(args...)
	unsafe_bit_map!(f, dest, args...)
end

checklengths(a) = nothing

@inline function checklengths(a, b)
	length(a) == length(b) || throw(DimensionMismatch("sizes of A and B must match"))
	nothing
end

@inline function checklengths(args...)
	allequal(length(arg) for arg in args) || throw(DimensionMismatch("all arguments must have the same length"))
	nothing
end


# 1-ary functions
function unsafe_bit_map!(f::F, dest::MBitString{L,N}, A::AbstractBitString{L,N}) where {F,L,N}
	destc = dest.chunks
	for i = 1:N
   	destc[i] = f(A.chunks[i])
   end
	destc[N] &= Base._msk_end(L)
	dest
end

# 2-ary functions
@inline function unsafe_bit_map!(f::F, dest::MBitString{L,N}, A::AbstractBitString{L,N}, B::AbstractBitString{L,N}) where {L,N,F}
	destc = dest.chunks
	@inbounds for i = 1:N
   	destc[i] = f(A.chunks[i], B.chunks[i])
   end
   @inbounds destc[N] &= Base._msk_end(L)
	dest
end


# n-ary functions
function unsafe_bit_map!(f::F, dest::MBitString{L,N}, args::AbstractBitString{L,N}...) where {L,N,F}
	destc = dest.chunks
	@inbounds for i = 1:N
   	destc[i] = f((arg.chunks[i] for arg in args)...)
   end
	@inbounds destc[N] &= Base._msk_end(L)
	dest
end


# ## Broadcasting
#
# # broadcastable(b::AbstractBitString) = b
# #
# # # associative bitwise functions
# # const BitwiseFun = Union{typeof(&), typeof(|), typeof(xor), typeof(~)}
# #
# #
# # # Create a custom style so that we can dispatch to our custom implementation
# # struct BitStringStyle{L,N} <: Broadcast.AbstractArrayStyle{1} end
# # BroadcastStyle(::Type{AbstractBitString{L,N}}) where {L,N} = BitStringStyle{L,N}()
# # BroadcastStyle(::BitStringStyle, t::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{max(1,N)}()
# #
# # # fallback -- default to ArrayStyle
# # broadcasted(bc::BitStringStyle, f::F, args...) where {F} = broadcasted(Broadcast.DefaultArrayStyle{1}(), f, args...)
# #
# # broadcasted(bc::BitStringStyle, f::BitwiseFun, args...) = bs_broadcast(f, args...)
# # broadcasted(bc::BitStringStyle, ::typeof(min), args...) = bs_broadcast(&, args...)
# # broadcasted(bc::BitStringStyle, ::typeof(max), args...) = bs_broadcast(|, args...)
# # broadcasted(bc::BitStringStyle, ::typeof(!=), a, b) = bs_broadcast(xor, a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(==), a, b) = bs_broadcast((x,y) -> ~xor(x,y), a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(^),  a, b) = bs_broadcast((x,y) -> x | ~y, a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(>), a, b) = bs_broadcast((x,y) -> x & ~y, a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(>=), a, b) = bs_broadcast((x,y) -> x | ~y, a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(<), a, b) = bs_broadcast((x,y) -> y & ~x, a, b)
# # broadcasted(bc::BitStringStyle, ::typeof(<=), a, b) = bs_broadcast((x,y) -> y | ~x, a, b)
# #
# # # Custom implementation.  For N=1, should take about 2ns (2μs per 1000)
# # function bs_broadcast(f::F, a::AbstractBitString{L,N}, b::AbstractBitString{L,N}) where {F,L,N}
# # 	mv = MVector{N,UInt64}(undef)
# # 	for i in 1:N
# # 		mv[i] = f(a.chunks[i], b.chunks[i])
# # 	end
# # 	AbstractBitString{L,N}(Tuple(mv))
# # end


## Linear algebra

@inline function dot(x::AbstractBitString, y::AbstractBitString)
    # simplest way to mimic Array dot behavior
    length(x) == length(y) || throw(DimensionMismatch())
    s = 0
    xc = x.chunks
    yc = y.chunks
    @inbounds for i = 1:length(xc)
        s += count_ones(xc[i] & yc[i])
    end
    s
end



# Just like dot, bit with xor instead of &
function hammdist(a::AbstractBitString, b::AbstractBitString)
	# simplest way to mimic Array dot behavior
	length(x) == length(y) || throw(DimensionMismatch())
	s = 0
	xc = x.chunks
	yc = y.chunks
	@inbounds for i = 1:length(xc)
		 s += count_ones(xc[i] ⊻ yc[i])
	end
	s
end


sum(b::AbstractBitString) = count(b)

function count(b::AbstractBitString)
	s = 0
	chk = b.chunks
	@inbounds for i = 1:length(chk)
		 s += count_ones(chk[i])
	end
	s
end


function parity(b::AbstractBitString)
	s = false
	chk = b.chunks
	@inbounds for i = 1:length(chk)
		 s ⊻= isodd(count_ones(chk[i]))
	end
	s
end


end	# module
