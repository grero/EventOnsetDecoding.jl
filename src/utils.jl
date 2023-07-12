MultivariateStats.toindices(label::AbstractVector{T}) where T <: Integer = label

get_cellnames(ppsth) = ppsth.cellnames

function get_fef_idx(ppsth, areas::Dict{String, Dict{String, String}})
	findall(is_fef,get_cellnames(ppsth))
end

function is_fef(cell::String,areas::Dict{String, Dict{String,String}})
	subject = NeuralAnalysis.DPHT.get_level_name("subject",cell)
	array = NeuralAnalysis.DPHT.get_level_name("array", cell)
	occursin("FEF", areas[subject][array])
end

function get_session_data(session::String, ppsth, trialidx, tlabel, rtimes, fef_idx::AbstractVector{Int64};rtime_min=100.0, rtime_max=300.0)
	all_sessions = DPHT.get_level_path.("session", get_cellnames(ppsth))
	sidx = findall(all_sessions[fef_idx].==session)
	if isempty(sidx)
		return nothing, nothing, nothing
	end
	_trialidx = trialidx[fef_idx[sidx]][1]
	_label = tlabel[fef_idx[sidx]][1]
	ntrials = length(_label)
	rtime = rtimes[session][_trialidx]
	rtidx = findall(rtime_min .< rtime .< rtime_max)
	rtime = rtime[rtidx]
	_label = _label[rtidx]
	X = ppsth.counts[:,rtidx,fef_idx[sidx]]
	bins = ppsth.bins
	binsize = bins[2] - bins[1]
	X, _label, rtime
end

function split_set(idx::Vector{T}, prop::Vector{Rational{T3}}) where T where T3 <: Integer
	offsets = fill(0, length(prop))
	ll = 0
	for (jj,q) in enumerate(prop)
		ddr = q*(length(idx)-ll)
		if isinteger(ddr)
			dd = Int64(ddr) + ll
		else
			dd = Int64(floor(ddr))
			ll = rem(ddr.num, ddr.den)
		end
		offsets[jj] = dd
	end
	offsets
end

"""
Splits the trials represented by `labels` into random subsets with the specified proportions
"""
function split_trials(labels::Vector{Vector{T2}},prop::Vector{Rational{T3}}=[1//2, 1//2];RNG=MersenneTwister(rand(UInt32))) where T2 where T3 <: Integer
	n = length(prop)
	out_labels = Vector{Vector{T2}}[]
	for (ii,label) in enumerate(labels)
		idx = shuffle(RNG, 1:length(label))
		olabel = Vector{T2}[]
		offsets = split_set(label, prop)
		offset = 0
		for dd in offsets
			push!(olabel, label[idx[offset+1:offset+dd]])
			offset += dd
		end
		push!(out_labels, olabel)
	end
	out_labels
end

"""
    sample_trials(X::Array{T,3}, label::Vector{T2}, testlabel=label;RNG=MersenneTwister(rand(UInt32)),
                                                                         ntrain=1500,
                                                                         ntest=100) where T <: Real where T2 <: Vector{T3} where T3

Sample `length(label) length(testlabel)` from the population count matrxi `X` by sampling trials with matching
labels for all cells independetly.
"""
function sample_trials(X::Array{T,3}, label::Vector{T2}, testlabel=label;return_indices=false,
																		RNG=MersenneTwister(rand(UInt32)),
                                                                         ntrain=1500,
                                                                         ntest=100) where T <: Real where T2 <: Vector{T3} where T3
    ncells,ntrials,nbins = size(X)
    traintestidx = split_trials([collect(1:length(l)) for l in label];RNG=RNG)
    trainidx = [idx[1] for idx in traintestidx]
    testidx = [idx[2] for idx in traintestidx]
    Xp = permutedims(X, [3,2,1])
	#TODO: This can be huge, maybe look into using a SparseArray here?
    Y = fill(0.0, nbins, ntrain+ntest, ncells)
    _trainidx, trainlabel = sample_trials!(view(Y, :, 1:ntrain,:), Xp, label, trainidx, RNG=RNG)
    _testidx, testlabel = sample_trials!(view(Y, :, (ntrain+1):(ntrain+ntest),:), Xp, testlabel, testidx, RNG=RNG)
	if return_indices
		return Y, trainlabel, testlabel, _trainidx, _testidx
	end
    Y, trainlabel, testlabel
end

function sample_trials!(Y::AbstractArray{T,3}, X::AbstractArray{T,3},labels::Vector{Vector{T2}}, gidx::Vector{Vector{Int64}};cellidx=1:size(Y,3),RNG=MersenneTwister(rand(UInt32))) where T <: Real where T2 <: Integer
	_,ntrials,ncells2 = size(Y)
    nbins, _, ncells = size(X)
    tidx = fill(0, ncells2, ntrials)
	flat_labels = cat(labels..., dims=1)
	out_label = fill(0, ntrials)
	for t in 1:ntrials
		#choose a random label
		label = rand(RNG, flat_labels)
		out_label[t] = label
		for (ci,c) in enumerate(cellidx)
			idxs = shuffle(RNG, gidx[c])
			lc = labels[c]
			for _idx in idxs
				if lc[_idx] == label
					Y[:,t,ci] .= X[:,_idx,c]
					tidx[ci,t] = _idx
					break
				end
			end
		end
	end
    tidx, out_label
end

function confusion_matrix(pp::Matrix{T}, Y::Vector{Int64}) where T <: Real
	ntrials, ncats = size(pp)
	Ξ = fill(0.0, ncats, ncats)
	nn = fill(0, 1, ncats)
	for k in 2:ntrials
		i = Y[k]
		j = argmax(pp[k,:])
		Ξ[j,i] += 1.0
		nn[i] += 1 
	end
	Ξ ./= nn
end