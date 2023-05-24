using MultivariateStats
using ProgressMeter
using CRC32c
using HDF5
using Random
using StatsBase
using DataProcessingHierarchyTools
const DPHT = DataProcessingHierarchyTools

struct DecoderArgs
	sessions::Vector{String}
	locations::AbstractVector{Int64}
	windows::AbstractVector{Float64}
	latencies::AbstractVector{Float64}
	nruns::Int64
	ntrain::Int64
	ntest::Int64
	sessionidx::AbstractVector{Int64}
	cellidx::AbstractVector{Int64}
	shuffle_bins::Bool
	remove_window::Union{Nothing, Tuple{Float64, Float64}}
	combine_locations::Bool
	combine_training_only::Bool
	save_full_performance::Bool
	save_weights::Bool
	save_projections::Bool
	shuffle_each_trial::Bool
	simple_shuffle::Bool
	restricted_shuffle::Bool
	use_new_decoder::Bool
	reverse_bins::Bool
	shuffle_latency::Bool
	shuffle_trials::Bool
	fix_resolution::Bool
	shift_forward::Bool
	difference_decoder::Bool
	shuffle_training::Bool
	at_source::Bool
	max_shuffle_latency::Float64
	proportional_shuffle::Float64
	in_target_space::Bool
	baseline_end::Float64
	rtime_min::Float64
	rtime_max::Float64
	mixin_postcue::Bool # should we include post-cue responses in the null-class
end

function DecoderArgs(sessions, locations,cellidx::AbstractVector{Int64};kvs...)
	latencies = get(kvs, :latencies, range(100, step=-10.0, stop=0.0))
	args = Any[]
	push!(args, sessions)
	push!(args, locations)
	push!(args, get(kvs, :windows, range(5.0, step=10.0, stop=50.0)))
	push!(args, latencies)
	push!(args, get(kvs, :nruns, 50))
	push!(args, get(kvs, :ntrain, 1500))
	push!(args, get(kvs, :ntest, 100))
	push!(args, get(kvs, :sessionidx, [1:length(sessions);]))
	push!(args, cellidx)
	push!(args, get(kvs, :shuffle_bins, false))
	push!(args, get(kvs, :remove_window, nothing))
	push!(args, get(kvs, :combine_locations, false))
	push!(args, get(kvs, :combine_training_only, false))
	push!(args, get(kvs, :save_full_performance, true))
	push!(args, get(kvs, :save_weights, true))
	push!(args, get(kvs, :save_projections, true))
	push!(args, get(kvs, :shuffle_each_trial, false))
	push!(args, get(kvs, :simple_shuffle, false))
	push!(args, get(kvs, :restricted_shuffle, false))
	push!(args, get(kvs, :use_new_decoder, false))
	push!(args, get(kvs, :reverse_bins, false))
	push!(args, get(kvs, :shuffle_latency, false))
	push!(args, get(kvs, :shuffle_trials, false))
	push!(args, get(kvs, :fix_resolution, false))
	push!(args, get(kvs, :shift_forward, false))
	push!(args, get(kvs, :difference_decoder, false))
	push!(args, get(kvs, :shuffle_training, true))
	push!(args, get(kvs, :at_source, false))
	push!(args, get(kvs, :max_shuffle_latency, maximum(latencies)))
	push!(args, get(kvs, :proportional_shuffle, NaN))
	push!(args, get(kvs, :in_target_space, false))
	push!(args, get(kvs, :baseline_end, -300.0))
	push!(args, get(kvs, :rtime_min, 100.0))
	push!(args, get(kvs, :rtime_max, 300.0))
	push!(args, get(kvs, :mixin_postcue, false))
	DecoderArgs(args...)
end

function DecoderArgs(args::DecoderArgs;kvs...)
	pargs = Any[]
	for k in fieldnames(DecoderArgs)
		push!(pargs, get(kvs, k, getfield(args, k)))
	end
	DecoderArgs(pargs...)
end

"""
Get the filename using a crc32c hash of the supplied arguments.
"""
function get_filename(args::DecoderArgs)
	h = CRC32c.crc32c(string(args.windows))
	h = CRC32c.crc32c(string(args.latencies), h)
	h = CRC32c.crc32c(string(args.nruns), h)
	if args.ntrain != 1500
		h = CRC32c.crc32c(string(args.ntrain), h)
	end
	if args.ntest != 100
		h = CRC32c.crc32c(string(args.ntest), h)
	end

	h = CRC32c.crc32c(string(args.sessionidx), h)
	h = CRC32c.crc32c(string(args.cellidx), h)
	h = CRC32c.crc32c(string(args.shuffle_bins), h)
	h = CRC32c.crc32c(string(args.locations), h)
	if args.remove_window != nothing
		h = CRC32c.crc32c(string(args.remove_window), h)
	end
	if args.combine_locations
		h = CRC32c.crc32c("combine_locations", h)
	end
	if args.combine_training_only
		h = CRC32c.crc32c("combine_training_only", h)
	end
	if args.save_full_performance == false
		h = CRC32c.crc32c("truncate_perforance", h)
	end
	if args.save_weights == false
		h = CRC32c.crc32c("no_weights", h)
	end
	if args.save_projections == false
		h = CRC32c.crc32c("no_projections", h)
	end
	if args.shuffle_bins
		if args.shuffle_each_trial == false
			h = CRC32c.crc32c("shuffle_once", h)
		end
		if args.simple_shuffle
			h = CRC32c.crc32c("simple_shuffle", h)
		end
		if args.restricted_shuffle
			h = CRC32c.crc32c("restricted_shuffle", h)
		end
	end
	if args.use_new_decoder
		h = CRC32c.crc32c("use_new_decoder", h)
	end
	if args.reverse_bins
		h = CRC32c.crc32c("reverse_bins", h)
	end
	if args.shuffle_latency
		if !args.shuffle_each_trial
			h = CRC32c.crc32c("shuffle_once", h)
		end
		h = CRC32c.crc32c("shuffle_latency", h)
		if args.fix_resolution
			h = CRC32c.crc32c("fix_resolution", h)
		end
		if args.shift_forward
			h = CRC32c.crc32c("shift_forward", h)
		end
		if args.max_shuffle_latency != maximum(args.latencies)
			h = CRC32c.crc32c("max_shuffle_latency = $(args.max_shuffle_latency)", h)
		end
		if isfinite(args.proportional_shuffle)
			h = CRC32c.crc32c("proportional_shuffle", h)
		end
	end
	if args.shuffle_trials
		h = CRC32c.crc32c("shuffle_trials", h)
	end
	if args.difference_decoder
		h = CRC32c.crc32c("difference_decoder", h)
	end
	if args.shuffle_bins | args.shuffle_latency
		if args.shuffle_training == false
			h = CRC32c.crc32c("shuffle_test_only", h)
		end
		if args.at_source
			h = CRC32c.crc32c("at_source", h)
		end
	end
	if args.in_target_space
		h = CRC32c.crc32c("in_target_space", h)
	end
	if args.baseline_end != -300.0
		h = CRC32c.crc32c(string(args.baseline_end), h)
	end
	if args.rtime_max != 300.0
		h = CRC32c.crc32c(string(args.rtime_max), h)
	end
	if args.rtime_min != 100.0
		h = CRC32c.crc32c(string(args.rtime_min), h)
	end
	if args.mixin_postcue
		h = CRC32c.crc32c(string(true), h)
	end

	qq = string(h, base=16)
	subject = lowercase(join(unique(DPHT.get_level_name.("subject", args.sessions)), '_'))
	fname = "$(subject)_rtime_pseudo_performance_distr_$(qq)_v7.hdf5"
	fname
end

"""
```
function run_rtime_decoder(ppsths, trialidx, tlabel, rtimes, args::DecoderArgs;redo=false,
							  index::Union{Nothing, Int64}=nothing,
							  decoder=MLJLinearModels.MultinomialRegression(0.5),
							  RNG::Union{Nothing, Vector{T2}}=nothing,
							  progress_tracker::Union{Nothing, Progress}=nothing,
							  num_cells::Union{Nothing, Int64}=nothing, do_save=true,
							  stop_task::Threads.Atomic{Bool}=Threads.Atomic{Bool}(false)) where T2 <: AbstractRNG
````
Train a decoder to predict the event that the spike counts in `ppsths` are aligned to. `ppsths` is a structure
containing the spike counts of cells for multiple trials across multiple sessions. `ppsths.counts` is a
nbins × ntrials × ncells array of spike counts, and `ppsths.cellnames` is a vector of strings with the full name of
each of the cell (e.g. Animal1/20140924/session01/array01/channel001/cell01), and thus `length(ppsths.cellnames) == size(ppsths.counts,3)`. `ppsths.bins`
indicates the time bins used to compute the spike counts.
"""
function run_rtime_decoder(ppsths, trialidx::Vector{Vector{Int64}}, tlabel::Vector{Vector{Int64}},
						   rtimes::Dict{String,Vector{Float64}}, args::DecoderArgs;redo=false,
							  index::Union{Nothing, Int64}=nothing,
							  decoder=MLJLinearModels.MultinomialRegression(0.5),
							  rseeds::Union{Vector{UInt32}, Nothing}=nothing,
							  RNGType::Type{T2}=MersenneTwister,
							  progress_tracker::Union{Nothing, Progress}=nothing,
							  num_cells::Union{Nothing, Int64}=nothing, do_save=true,
							  stop_task::Threads.Atomic{Bool}=Threads.Atomic{Bool}(false)
							  ) where T2 <: AbstractRNG

	if rseeds === nothing
		rseeds = rand(UInt32, args.nruns)
	else
		length(rseeds) == args.nruns || error("Please supply one RNG per run")
	end
	RNGs = [RNGType(r) for r in rseeds]
	if num_cells != nothing
		h = CRC32c.crc32c(string(num_cells), h)
	end
	if args.use_new_decoder | args.difference_decoder
		nc = 2
	else
		nc = 3
	end

	fname = get_filename(args)
	if !isdir("data")
		mkdir("data")
	end
	fname = joinpath("data",fname)
	if HDF5.ishdf5(fname) && redo == false
		pr = h5read(fname, "perf")
		rr = h5read(fname, "positive_rate")
		f1score = h5read(fname, "f1score")
	else
		fef_idxs = args.cellidx
		if num_cells != nothing
			num_cells = min(num_cells, length(fef_idxs))
			fef_idxs = shuffle(fef_idxs)[1:num_cells]
		end
		if args.reverse_bins
			bins = -1*reverse(ppsths.bins)[ppsths.windowsize+1:end]
			# since we have reversed the bins, the baseline for cue aligned responses now is after 0
			bidx = findall(maximum(args.windows) .< bins .< -args.baseline_end - maximum(args.windows))
		else
			bins = ppsths.bins
			bidx = findall(bins[1] + maximum(args.windows) .< bins .< args.baseline_end)
		end
		binsize = bins[2] - bins[1]

		# create a sub-sample of ppsth with the correct reaction times
		Xtot = fill(0.0, size(ppsths.counts,1), size(ppsths.counts, 2), length(fef_idxs))
		label_tot = Vector{Vector{Int64}}(undef, length(fef_idxs))
		celloffset = 0
		if args.shuffle_latency
			binsize = bins[2] - bins[1]
			l1 = round(Int64, args.max_shuffle_latency/binsize)
			l0 = round(Int64, minimum(args.latencies)/binsize)
			l0 = max(l0, 1)
			if !args.shift_forward
				l0 = -l1
				l1 = -l0
			end
		end
		for i in args.sessionidx
			X, _label, _rtime = get_session_data(i,ppsths, trialidx, tlabel, rtimes, fef_idxs;rtime_min=args.rtime_min,
												 											  rtime_max=args.rtime_max)
			if args.reverse_bins
				X .= X[end:-1:1, :,:]
			end
			if args.shuffle_latency & args.shuffle_each_trial & args.at_source
				for i in 1:size(X,2)
					Δ = rand(RNGs[1], l0:l1)
					for j in 1:size(X,3)
						X[:,i,j] = circshift(X[:,i,j], Δ)
					end
				end
			end
			if X == nothing
				continue
			end
			# TODO: Make use of LocationSpec here 
			ttidx = findall(in(args.locations), _label)
			_label = [findfirst(args.locations.==l) for l in _label[ttidx]]
			X = X[:, ttidx,:]
			_rtime = _rtime[ttidx]
			if args.combine_locations
				fill!(_label, 1)
			end
			_ncells = size(X,3)
			cidx = celloffset+1:celloffset+_ncells
			Xtot[:, 1:size(X,2),cidx] .= X
			for (jj, cc) in enumerate(cidx)
				label_tot[cc] = _label
			end
			celloffset += _ncells
		end
		Xtot = Xtot[:,:,1:celloffset]
		label_tot = label_tot[1:celloffset]
		ncells = size(Xtot,3)
		ulabel = union(map(unique, label_tot)...)
		use_locations = ulabel
		sort!(use_locations)
		nlocations = length(use_locations)
		# make sure we are not trying to use locations that are not present in the data
		# ulabel = intersect(locations, ulabel)
		sort!(ulabel)
		nlabels = maximum(ulabel)
		if args.combine_training_only
			@info "Combining training..."
		end
		#TODO: Save directly to file here so that we can inspect as we go
		if args.remove_window != nothing
			rwidx0 = searchsortedfirst(bins, args.remove_window[1])
			rwidx1 = searchsortedlast(bins, args.remove_window[2])
		else
			rwidx0 = 2
			rwidx1 = 1
		end
		if progress_tracker == nothing
			prog = Progress(args.nruns*length(args.windows)*length(args.latencies)*nlocations)
		else
			prog = progress_tracker
		end
		Xtot2 = Xtot
		if args.at_source
			# do whatever shuffling we are supposed to do at the source
			if args.shuffle_latency & !args.shuffle_each_trial
				Δ = rand(RNGs[1], l0:l1)
				Xtot2 = circshift(Xtot, (Δ, 0, 0))
			end
		end
		rr = fill(0.0, length(args.windows), length(args.latencies), nlocations,args.nruns)
		ξ = fill(0.0, nc,nc, length(args.windows), length(args.latencies), nlocations,args.nruns)
		# variables to hold the projections from cells to subspace
		proj = fill(0.0, ncells,nc-1, length(args.windows), length(args.latencies), nlocations,args.nruns)
		ndims = fill(0.0, length(args.windows), length(args.latencies), nlocations,args.nruns)
		# the means of each category in the subspace
		pmeans = fill(0.0, ncells,nc, length(args.windows), length(args.latencies), nlocations, args.nruns)
		pcameans = fill(0.0, ncells, length(args.windows), length(args.latencies), nlocations, args.nruns)
		# to coordinate the writes
		sl = Threads.ReentrantLock()
		HDF5.h5open(fname,"w") do fid
			fid["window"] = collect(args.windows)
			fid["latency"] = collect(args.latencies)
			fid["nruns"] = args.nruns
			fid["sessionidx"] = [args.sessionidx;]
			fid["rseeds"] = rseeds
			fid["RNGType"] = String(Symbol(RNGType)) 
			fid["locations"] = use_locations
			fid["bins"] = collect(bins)
			fid["remove_window"] = args.remove_window != nothing ? [args.remove_window...] : [0.0, 0.0]

			if args.save_full_performance
				pr = fill(0.0, size(Xtot,1), length(args.windows), length(args.latencies), nlocations, args.nruns)
				perf = create_dataset(fid, "perf",HDF5.datatype(Float64), dataspace(size(Xtot,1), length(args.windows), length(args.latencies), nlocations, args.nruns), chunk=(size(Xtot,1), length(args.windows), length(args.latencies), nlocations, 1))
				posterior = fill(0.0, size(Xtot,1), nc, length(args.windows), length(args.latencies), nlocations, args.nruns)
				posterior_ = create_dataset(fid, "posterior", HDF5.datatype(Float64), dataspace(size(Xtot,1), nc, length(args.windows), length(args.latencies), nlocations, args.nruns), chunk=(size(Xtot,1), nc, length(args.windows), length(args.latencies), nlocations, 1))
				entropy = fill(0.0, size(Xtot,1), length(args.windows), length(args.latencies), nlocations, args.nruns)
				entropy_ = create_dataset(fid, "entropy", HDF5.datatype(Float64), dataspace(size(Xtot,1), length(args.windows), length(args.latencies), nlocations, args.nruns), chunk=(size(Xtot,1), length(args.windows), length(args.latencies), nlocations, 1))
			else
				# only save performance at the trained window; mainly useful in conjunction with shuffling
				pr = fill(0.0, 1, length(args.windows), length(args.latencies), nlocations, args.nruns)
				posterior = fill(0.0, 1, nc, length(args.windows), length(args.latencies), nlocations, args.nruns)
				entropy = fill(0.0, 1, length(args.windows), length(args.latencies), nlocations, args.nruns)
			end
			f1score = fill(0.0, length(args.windows), length(args.latencies), nlocations,args.nruns)
			f1score_ = create_dataset(fid, "f1score", HDF5.datatype(Float64), dataspace(length(args.windows), length(args.latencies), nlocations, args.nruns), chunk=(length(args.windows), length(args.latencies), nlocations, 1))

			max_perf = Threads.Atomic{Float64}(0.0)
			σ_perf = Threads.Atomic{Float64}(0.0)
			Threads.@threads for r in 1:args.nruns
			#for r in 1:nruns
				if stop_task[]
					@info "Stopping thread $(Threads.threaid) of $(Threads.nthreads())"
					break
				end
				qrng = RNGs[r]
				# private copy for each thread so that we can shuffle it
				# TODO: Include post-cue response as well.
				Yt, train_label,test_label =  sample_trials(permutedims(Xtot2, [3,2,1]), label_tot;RNG=qrng,ntrain=args.ntrain, ntest=args.ntest)
				@assert size(Yt,1) == size(Xtot2, 1)
				ntrain, ntest = (length(train_label), length(test_label))
				# if we are shuffling bins, we need to keep a copy of the original data
				if !args.at_source & (args.shuffle_bins | args.shuffle_latency)
					Yt2 = fill!(similar(Yt), 0.0)
					Yt2 .= Yt
				else
					Yt2 = Yt
				end
				nl = fill(0, nlabels)
				nli = fill(0, nlabels)
				for i in 1:ntrain
					li = train_label[i]
					nl[li] += 1
				end
				nlx = maximum(nl)
				Xr = fill(0.0, nc*nlx, size(Yt,3), nlabels)  # matrix to thold the reaction time decoding values.
				Yr = fill(0, nc*nlx, nlabels) # class labels
				Ytest = permutedims(Yt[:, ntrain+1:end,:], [3,2,1])
				# replace window with data from baseline
				# this is probably not the best approach; the baseline window could be even more different, making the decoder even better at predicting
				# rather, replace with the previous window
				ww = rwidx1-rwidx0 + 1
				Ytest[:,:, rwidx0:rwidx1] .= Ytest[:,:, rwidx0-ww:rwidx1-ww]
				Ytest2 = fill!(similar(Ytest), 0.0)

				for (wi,window) in enumerate(args.windows)
					t1 = -window
					if args.restricted_shuffle
						t1 = 0.0
						if args.reverse_bins
							t0 = -maximum(args.latencies) - window
						else
							t0 = -maximum(args.latencies) - window
						end
					else
						t0 = args.baseline_end
					end
					eeidx = findall(t0 .<= bins .<= t1)
					fill!(Ytest2, 0.0)
					Ytest2 .= Ytest
					testbins = bins
					if !args.difference_decoder
						for i1 in 1:size(Ytest,3)
							idx0 = i1
							idx1 = searchsortedfirst(bins, bins[i1] + window)-1
							for i2 in 1:size(Ytest,2)
								for i3 in 1:size(Ytest,1)
									uu = 0.0
									for iu in idx0:idx1
										uu += Ytest2[i3,i2,iu]
									end
									Ytest2[i3,i2,i1] = uu
								end
							end
						end
					end
					eeidx2 = fill!(similar(eeidx), 0)
					eeidx2 .= eeidx
					if !args.at_source & (args.shuffle_bins | args.shuffle_latency)
						if args.shuffle_latency
							if args.fix_resolution
								binsize = bins[2] - bins[1]
								if isfinite(args.proportional_shuffle)
									w = round(Int64, window/binsize)
									l1 = round(Int64, args.proportional_shuffle*window/binsize)
									l0 = w
								else
									l1 = round(Int64, args.max_shuffle_latency/binsize)
									l0 = round(Int64, minimum(args.latencies)/binsize)
									l0 = max(l0, 1)
									w = 1
								end

								if args.shift_forward
									Δidxtt = l0:w:l1
								else
									Δidxtt = -l1:w:-l0
								end
							else
								qidx0 = searchsortedfirst(bins, -maximum(args.latencies))
								qidx1 = searchsortedlast(bins, -minimum(args.latencies))
								Δidx = div(qidx1 - qidx0 + 1,2)
							end
						end
						sidx = rebin(eeidx, round(Int64, window/binsize))
						# shuffle the pre-movement bins for each trial
						if args.shuffle_each_trial
							for tt in 1:ntrain
								if args.shuffle_training
									if args.simple_shuffle
										shuffle!(qrng, eeidx2)
									elseif args.shuffle_latency
										Δ = rand(qrng, Δidxtr)
										y = circshift(Yt[:, tt,:], (Δ,0))
										Yt2[:,tt,:] .= y
									else
										shuffle_with_precision!(qrng, eeidx2, eeidx, sidx)
									end
								end
								#Yt2[eeidx,tt,:] .= Yt[eeidx2, tt,:]
							end
							for tt in 1:ntest
								if args.shuffle_latency
									Δ = rand(qrng, Δidxtt)

									# shift everything by an amount Δ
									y = circshift(Ytest2[:,tt,:], (0, Δ))
									Ytest2[:,tt,:] .= y
								else
									if args.simple_shuffle
										shuffle!(qrng, eeidx2)
									else
										shuffle_with_precision!(qrng, eeidx2, eeidx, sidx)
									end
									Ytest2[:,tt,eeidx] .= Ytest2[:,tt,eeidx2]
								end

							end
						else
							if args.shuffle_latency
								Δ = rand(qrng, Δidxtt)
								Ytest2 .= circshift(Ytest2, (0,0,Δ))
								if args.shuffle_training
									Δ = rand(qrng, Δidxtr)
									Yt2 .= circshift(Yt, (Δ,0, 0))
								end
							else
								if args.simple_shuffle
									shuffle!(qrng, eeidx2)
								else
									shuffle_with_precision!(qrng, eeidx2, eeidx, sidx)
								end
								# TODO: Don't shuffle Yt2 here
								#Yt2[eeidx,:,:] .= Yt[eeidx2, :,:]
								Ytest2[:,:,eeidx] .= Ytest2[:,:,eeidx2]
							end
						end

					end

					# bidx = findall(bins[1] + window .< bins .< args.baseline_end)
					for (li, latency) in enumerate(args.latencies)
						if args.reverse_bins
							eidx = findall(-latency .<= bins .< -latency + window)
							pidx = findall(-latency - window .<= bins .< -latency)
							ppidx = findall(-latency - 2*window .<= bins .< -latency - window)
						else
							eidx = findall(-latency - window .<= bins .< -latency)
							# the window immediately preceding the above window
							pidx = findall(-latency - 2*window .<= bins .< -latency - window)
							ppidx = findall(-latency - 3*window .<= bins .< -latency - 2*window)
						end
						Xpp = permutedims(dropdims(mean(Yt2[eidx, 1:ntrain, :],dims=1), dims=1), [2,1])
						if args.in_target_space
							_pca = MultivariateStats.fit(MultivariateStats.PCA, Xpp)
							yp = predict(_pca, Xpp)
							_lda = MultivariateStats.fit(MultivariateStats.MulticlassLDA, nlabels, yp, train_label)
							# project both training and testing data
							# training
							np = size(_lda.proj,2)
							for i2 in axes(Yt2,2)
								for i1 in axes(Yt2,1)
									Yt2[i1,i2,1:np] = predict(_lda, predict(_pca, Yt2[i1,i2,:]))
								end
							end
							for i3 in axes(Ytest2,3)
								for i2 in axes(Ytest2,2)
									Ytest2[1:np, i2,i3] = predict(_lda, predict(_pca, Ytest2[:,i2,i3]))
								end
							end
							Ytest3 = view(Ytest2, 1:np, :, :)
							Yt3 = view(Yt2, :, :, 1:np)
							Xr3 = view(Xr, :, 1:np, :)
						else
							Ytest3 = Ytest2
							Yt3  = Yt2
							Xr3 = Xr
						end
						# TODO: How should we shuffle this?
						# this could change
						qidx = 0.0 .< bins .<= window
						midx = first(eidx)
						# optionally, draw these bins from some other period
						# random bins from the baseline
						neidx = length(eidx)
						fidx = bidx[(bidx .+ (neidx-1) .<= maximum(bidx)).&(bidx .- neidx .> 0)]
						if length(fidx) == 0
							error("No baseline avilable. bidx=$(bidx) neidx=$(neidx)")
						end
						fill!(nli, 1)
						for it in 1:ntrain
							if !args.at_source & args.shuffle_each_trial
								shuffle!(qrng, eeidx2)
							end
							shuffle!(qrng, fidx)
							# make sure we are only using valid points for the slope
							ll = train_label[it]
							nn = nl[ll]
							iq = nli[ll]
							if !args.at_source & args.shuffle_bins
								# find the new bin at the position of midx after shuffling
								_eidx = midx - eeidx[1] + 1
								_eidx = range(eeidx2[_eidx], step=1, length=neidx)
							else
								_eidx = eidx
							end
							Xr3[iq,:,ll] .= dropdims(mean(Yt3[_eidx, it, :],dims=1), dims=1)
							if !args.at_source & args.shuffle_bins
								if args.use_new_decoder
									_qidx = (_eidx[1]-neidx):(_eidx[1]-1)
									Xr3[nn+iq,:,ll] = dropdims(mean(Yt3[_qidx, it, :],dims=1), dims=1)
								end
							else
								if args.difference_decoder
									Xr3[iq,:,ll] .-= dropdims(mean(Yt3[pidx, it, :],dims=1), dims=1)
								end
								if args.use_new_decoder
									Xr3[nn+iq,:,ll] = dropdims(mean(Yt3[pidx, it, :],dims=1), dims=1)
								elseif args.difference_decoder
									# use random slope from the baseline
									Xr3[nn+iq,:,ll] = dropdims(mean(Yt3[fidx[1]:fidx[1]+neidx-1, it, :],dims=1), dims=1)-dropdims(mean(Yt3[fidx[1]-neidx:fidx[1]-1, it, :],dims=1), dims=1)
								end
							end
							if !(args.use_new_decoder | args.difference_decoder)
								Xr3[nn+iq,:,ll] .= dropdims(mean(Yt3[qidx, it, :], dims=1), dims=1)
								Xr3[2*nn+iq,:,ll] .= dropdims(mean(Yt3[fidx, it, :], dims=1), dims=1)
								Yr[iq,ll] = 2
								Yr[nn+iq,ll] = 1
								Yr[2*nn+iq,ll] = 3
							else
								Yr[iq,ll] = 2
								Yr[nn+iq,ll] = 1
							end
							nli[ll] += 1
						end

						# if we are combining locations, let's just concatenate first, and set use_locations as [1].
						# actually, if we are combining locations, we don't need the location label at all, and so we don't even need to sample from the pseudo-population. Just grab the relevant window and combine across all cells. Likewise, for training, just project the full PSTH.
						# TODO: Combine locations here instead
						if args.combine_training_only
							Yr2 = Array{Int64,2}(undef, 0,1)
							Xr5 = Array{Float64,3}(undef, 0, size(Xr3,2),1)
							for (il,location) in enumerate(use_locations)
								nll = nl[location]
								Yr2 = cat(Yr2, Yr[1:nc*nll, location], dims=1)
								Xr5 = cat(Xr5, Xr3[1:nc*nll, :, location], dims=1)
							end
							training_locations = [1]
						else
							training_locations = use_locations
							Yr2 = Yr
							Xr5 = Xr3
						end
						Wp = Vector{MultivariateStats.MulticlassLDA{Float64}}(undef, length(use_locations))
						pcap = Vector{MultivariateStats.PCA{Float64}}(undef, length(use_locations))
						for (il,location) in enumerate(training_locations)
							nll = nl[location]
							if nll == 0
								error("No trials found with label location")
							end
							if (length(unique(Yr2[1:nc*nll, location])) < 2) || (minimum(Yr2[1:nc*nll, location]) == 0)
								error("Missing category labels")
							end
							if !args.in_target_space
								# if we are in target space, we have already done PCA
								_pca = MultivariateStats.fit(PCA, permutedims(Xr5[1:nc*nll,:,location],[2,1]))
								Xrp = permutedims(MultivariateStats.predict(_pca, permutedims(Xr5[1:nc*nll, :, location], [2,1])), [2,1])

								W,p = train_decoder(decoder, Xrp, Yr2[1:nc*nll, location])
								proj[:,1:size(W.proj,2), wi, li, il, r] .= _pca.proj*W.proj
								pcameans[:,wi, li, il, r] .= _pca.mean
								if args.combine_training_only
									for rri in 1:length(Wp)
										Wp[rri] = W
										pcap[rri] = _pca
									end
								else
									Wp[il] = W
									pcap[il] = _pca
								end
							else
								W,p = train_decoder(decoder, Xr5[1:nc*nll, :, location], Yr2[1:nc*nll, location])
								proj[1:size(W.proj,1),1:size(W.proj,2), wi, li, il, r] .= W.proj
								if args.combine_training_only
									for rri in 1:length(Wp)
										Wp[rri] = W
									end
								else
									Wp[il] = W
								end
							end
							ndims[wi, li, il, r] = size(W.proj,2)
							pmeans[1:size(W.proj,2), :, wi, li,il,r] .= W.pmeans
							ξ[:,:,wi, li, il, r] .= confusion_matrix(p, Yr2[1:nc*nll, location])
							t0 = time()
						end
						for (il,location) in enumerate(use_locations)
							W = Wp[il]
							if isassigned(pcap, il)
								_pca = pcap[il]
							end
							lidx = findall(test_label.==location)
							#project onto a test trial
							wl = round(Int64, window/binsize)
							if args.difference_decoder
								# TODO: IS this a problem?
								Q = fill(0, size(Ytest3,3)-2*wl, length(lidx))
							else
								Q = fill(0, size(Ytest3,3), length(lidx))
							end
							y = fill(0.0, size(Ytest3,1))
							midx = searchsortedfirst(testbins[1+wl:end], bins[midx])
							for i in 1:size(Q,2)
								for j in 1:size(Q,1)
									#y = MultivariateStats.predict(_pca,Ytest2[:,lidx[i],j])
									#q = [y;1.0]'*W
									if args.difference_decoder
										#TODO: Include pre-cue here
										y .= sum(Ytest3[:,lidx[i], j+wl:j+2*wl-1],dims=2) - sum(Ytest3[:,lidx[i], j:j+wl-1],dims=2)
										if args.in_target_space
											q = get_posterior(W, y)
										else
											q = get_posterior(_pca, W, y)
										end
									else
										if args.in_target_space
											q = get_posterior(W, Ytest3[:,lidx[i],j])
										else
											q = get_posterior(_pca, W, Ytest3[:,lidx[i],j])
										end
									end
									q .= exp.(q)
									#q ./= sum(q)
									# make positive
									#q .= (q .- minimum(q))./(maximum(q) - minimum(q))
									# normalize
									q ./= sum(q)
									if args.save_full_performance
										posterior[j,:,wi,li,il,r] .+= q
										entropy[j,wi,li,il,r] += -sum(q.*log2.(q))
									else
										if j == midx
											posterior[1, :, wi,li,il,r] .+= q
											entropy[1,wi,li,il,r] += -sum(q.*log2.(q))
										end
									end
									Q[j,i] = argmax(q[:])
								end
							end
							# compute mean posterior
							posterior[:,:,wi,li,il,r] ./= size(Q,2)
							entropy[:,wi,li,il,r] ./= size(Q,2)
							#@show midx, size(Ytest2), size(Q), window, latency
							tp = sum(Q[midx,:].==2)
							blidx0 = searchsortedfirst(testbins, -500.0)
							blidx1 = searchsortedfirst(testbins, -400.0)
							fp =  sum(mean(Q[blidx0:blidx1,:].==2,dims=1))
							fn = sum(Q[midx,:].!=2)
							f1score[wi,li,il,r] = tp/(tp + 0.5*(fp+fn))
							_perf = fill(0.0, size(Q,1))
							if args.mixin_postcue
								np = 0
								nn = 0
								# balance testing here; for each time point, include some baseline activity
								qb  = 0.0
								fp = fill!(similar(_perf), 0.0)
								fn = fill!(similar(_perf), 0.0)
								for tt in 1:size(Q,2)
									# grab a random bin from the baseline
									qv = rand()
									if qv < 0.5
										_bidx = rand(bidx)
										fp .+= Q[_bidx,tt] == 2
										nn += 1
									else
										_perf .+= Q[:,tt] .== 2
										fn .+= Q[:, tt] .== 1
										np += 1
									end
								end
								_perf ./= np
								fn ./= np
								fp ./= nn
								f1score[wi,li,il,r] = _perf[midx]/(_perf[midx] + 0.5*(fp[midx]+fn[midx]))
							else
								_perf .= dropdims(sum(Q.==2,dims=2),dims=2)./size(Q,2)
							end
							if args.save_full_performance
								pr[1+wl:1+wl+size(Q,1)-1,wi, li, il, r] .= _perf
								cpr = pr[midx, wi, li, il, r]
								fpr = mean(pr[blidx0:blidx1, wi, li,il, r])
							else
								pr[1,wi, li, il, r] = _perf[midx]
								cpr = pr[1, wi, li, il, r]
								fpr = mean(sum(Q[blidx0:blidx1,:].==2,dims=2)./size(Q,2))
							end
							rr[wi,li,il, r] = (cpr - fpr)/(cpr + fpr)
							Threads.atomic_max!(max_perf, maximum(pr[:,:,:,:,r]))
							Threads.atomic_max!(σ_perf, std(pr[:,:,:,:,r]))
							next!(prog; showvalues=[(:run, index), (:max_perf, max_perf[]),(:std, σ_perf[])])
						end
					end

				end
				# make sure the writes are clean
				lock(sl)
				perf[:,:,:,:,r] = pr[:,:,:,:,r]
				posterior_[:,:,:,:,:,r] = posterior[:,:,:,:,:,r]
				entropy_[:,:,:,:,r] = entropy[:,:,:,:,r] 
				f1score_[:,:,:,r] = f1score[:,:,:,r]
				flush(fid)
				unlock(sl)
			end
		end
		if do_save
			h5open(fname, "r+") do fid
				fid["confusion_matrx"] = ξ
				fid["positive_rate"] = rr
				if args.save_weights
					fid["weights"] = proj
				end
				if args.save_projections
					fid["pmeans"] = pmeans
					fid["pcameans"] = pcameans
					fid["ndims"] = ndims
				end
			end
		end
	end
	pr, rr, f1score, fname
end

function aggregator!(func::Function, Q::Matrix{Int64}, Ytest::Array{Float64,3})
	for i in 1:size(Q,2)
		for j in 1:size(Q,1)
			q = func(Ytest[:,i,j])
			Q[j,i] = argmax(q)
		end
	end
	Q
end

function aggregator!(func::Function, Q::Matrix{Int64}, Ytest::Array{Float64,3}, wl::Int64)
	y = fill(0.0, size(Ytest,1))
	for i in 1:size(Q,2)
		for j in 1:size(Q,1)
			for l in 1:size(Ytest,1)
				y[l] = 0.0
				d1 	= 0
				d0 = 0
				for k in 1:wl
					d1 += Ytest[l,i, j+wl+k-1]
					d0 += Ytest[l,i, j+k-1]
				end
				y[l] = d1 - d0
			end
			q = func(y)
			Q[j,i] = argmax(q)
		end
	end
	Q
end

"""
Return the number of surrogates already completed for the specified arguments
"""
function check_shuffle_progress(dargs::DecoderArgs, nshuffles::Int64)
	#dargs2 = DecoderArgs(dargs, save_full_performance=false, save_weights = false, save_projections=false)
	fname = joinpath("data",get_filename(dargs))
	fname = replace(fname, ".hdf5" => "_$(nshuffles)_surrogate.hdf5")
    perf = HDF5.h5open(fname) do fid
        read(fid, "perf")
    end
    idx = findall(maximum(perf, dims=(1,2,3)).>0.0)
    icounts = length(idx)
    icounts
end

function shuffle_decoders(nshuffles::Int64, args...;reuse=false, kvs...)
	dargs1 = args[end]
	dargs = DecoderArgs(dargs1, save_full_performance=false, save_weights = false, save_projections=false)
	labels = args[3]
	locations = dargs.locations
	ulabel = union(map(unique, labels)...)
	use_locations = intersect(locations, ulabel)
	if dargs.combine_locations | dargs.in_target_space
		nlocations = 1
	else
		nlocations = length(use_locations)
	end
	windows = dargs.windows
	latencies = dargs.latencies
	nruns = dargs.nruns
	redo = kvs[:redo]
	fname = get_filename(dargs1)
	fname = replace(fname, ".hdf5" => "_$(nshuffles)_surrogate.hdf5")
    fname_prog = "$(fname).inprogress"
    if isfile(fname_prog)
        error("It looks like the file $fname is in use by another process. If this is not the case, please delete the file $fname_prog and try again")
    end
	if reuse
		# attempt to find another file with identical arguments, but with lower number of shuffles
		fnames = replace(fname, ".hdf5" => "_*_surrogate.hdf5")
		files = glob(fnames)
		_nshuffles = 0
		for f in files
			ns = parse(Int64, split(split(f,'.')[1], '_')[end])
			if nsuffles > ns > _snuffles
				_nshuffles = ns
			end
		end
		if _nshuffles == 0
			@warn "No files with number of shuffles <= $nsuffles was found. Starting from scratch"
		else
			#TODO: Read the contents of the previous file into a new file, resizing the perf and f1 arrays as necessary

		end	
			
	end
    icounts = 0
    mode = "w"
    if HDF5.ishdf5(fname) && !redo
        # check if we completed the file
        HDF5.h5open(fname) do fid
            if "perf" in keys(fid)
                perf = read(fid,"perf")
                idx = findall(maximum(perf, dims=(1,2,3)).>0.0)
                icounts = length(idx)
                if icounts < nshuffles
                    mode = "r+"
                    redo = true
                end
            end
        end
    end
	N = (nshuffles-icounts)*nruns*length(windows)*length(latencies)*nlocations
	prog = Progress(N)
	stop_task = Threads.Atomic{Bool}(false)
	if !HDF5.ishdf5(fname) || redo
		HDF5.h5open(fname, mode) do fid
            if !("windows" in keys(fid))
                fid["windows"] = [windows;]
            end
            if !("latencies" in keys(fid))
                fid["latencies"] = [latencies;]
            end
            if !("perf" in keys(fid))
                perf = create_dataset(fid, "perf", HDF5.datatype(Float64), dataspace(length(windows), length(latencies), nlocations, nshuffles),chunk=(length(windows), length(latencies), nlocations, 1))
			end
			if !("posterior") in keys(fid)
                posterior = create_dataset(fid, "posterior", HDF5.datatype(Float64), dataspace(length(windows), length(latencies), nlocations, nshuffles),chunk=(length(windows), length(latencies), nlocations, 1))
            else
                perf = fid["posterior"]
            end
            if !("f1score" in keys(fid))
                f1score = create_dataset(fid, "f1score", HDF5.datatype(Float64), dataspace(length(windows), length(latencies), nlocations, nshuffles),chunk=(length(windows), length(latencies), nlocations, 1))
            else
                f1score = fid["f1score"]
            end
            try
                touch(fname_prog)
                for i in (icounts+1):nshuffles
                    pr, rr, f1 = run_rtime_decoder(args[1:end-1]..., dargs;kvs...,progress_tracker=prog, do_save=false,  index=i, stop_task=stop_task)
                    perf[:,:,:,i] = dropdims(mean(pr, dims=5), dims=(1,5))
                    f1score[:,:,:,i] = dropdims(mean(f1, dims=4), dims=4)
                    flush(fid)
                end
            catch ee
				if isa(ee, InterruptException)
					@info "Attempting to interrupt threads..."
					stop_task[] = true
				else
					rethrow(ee)
				end
            finally
                rm(fname_prog)
            end
		end
	end
	HDF5.h5open(fname) do fid
		read(fid, "perf"), read(fid, "f1score")
	end
end

function get_posterior(_pca, W::Matrix{T}, X::Array{T,3}, lidx::Vector{Int64}) where T <: Real
	Q = fill(0, size(X,1), min(length(lidx), 100))
	Xp = fill(0.0, size(X,1)+1)
	for i in 1:size(Q,2)
		lidxi = lidx[i]
		for j in 1:size(Q,1)
			Xp[1:end-1] .= MultivariateStats.predict(_pca,X[:,lidxi,j])
			q = Xp'*W
			Q[j,i] = argmax(q[:])
		end
	end
	pr
end

"""
Get the posterior when using regression, in which the projection onto the matrix `W` is analogous to the posterior
"""
function get_posterior(W::Matrix{Float64}, X::AbstractVector{Float64})
	MultivariateStats.predict(W, X)
end

"""
Get the posterior equivalent for a trained LDA, which we have to first compute the distance to each class
"""
function get_posterior(pca, lda::MultivariateStats.MulticlassLDA, X::AbstractVector{Float64})
	y = MultivariateStats.predict(pca, X)
	q = MultivariateStats.predict(lda, y)
	d = -dropdims(sum(abs2, q .- lda.pmeans, dims=1),dims=1)
end

function get_posterior(lda::MultivariateStats.MulticlassLDA, X::AbstractVector{Float64})
	q = MultivariateStats.predict(lda, X)
	d = -dropdims(sum(abs2, q .- lda.pmeans, dims=1),dims=1)
end

function train_decoder(decoder::Type{MultivariateStats.MulticlassLDA}, X::Matrix{T}, Y::Vector{Int64}) where T <: Real
	ncats = maximum(Y)
	ntrials,nvars = size(X)
	Xp = permutedims(X,[2,1])
	lda = MultivariateStats.fit(MultivariateStats.MulticlassLDA, Xp, Y)
	Xpp = MultivariateStats.predict(lda, Xp)
	pp = fill(0.0, ncats, ntrials)
	for k in 1:ntrials
		for i in 1:ncats
			d = sum(abs2, Xpp[:,k] - lda.pmeans[:,i])
			pp[i,k] = exp(-d)
		end
	end
	lda, permutedims(pp, [2,1])
end
