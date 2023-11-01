using EventOnsetDecoding
using EventOnsetDecoding: MultivariateStats
using StableRNGs
using StatsBase
using Test


@testset begin "Basic"
    rng = StableRNG(1234)
    # a highly simplistic model
    bins = [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,10.0,15.0]
    nbins = length(bins)
    nt = 100
    ncells = 5
    cellnames = ["animalZ/20140904/session01/array01/channel$(string(i,pad=3))/cell01" for i in 1:ncells]
    μ = 5.0*randn(rng, 1,ncells)
    X = 0.1*randn(rng, nbins,nt, ncells)

    X[8,:,:] .+= μ 
    X[9,:,:] .+= 0.75*μ
    X[10,:,:] .+= 0.5*μ
    trialidx = [[1:nt;] for i in 1:ncells]
    tlabel = [fill(1,nt) for i in 1:ncells]
    rtimes = Dict("animalZ/20140904/session01"=>fill(0.1, nt)) 
    rseeds = UInt32[0xaa815070, 0xcb646653, 0x584bf898, 0xd099425a, 0x20e80af4]
    dargs = EventOnsetDecoding.DecoderArgs(["animalZ/20140904/session01"],[1],1:ncells, windows=[5.0], latencies=[0.0], 
                                            difference_decoder=true,rtime_min=0.0, rtime_max=1.0,
                                            baseline_end=-10.0,nruns=5, mixin_postcue=true, save_sample_indices=false)

    fname = EventOnsetDecoding.get_filename(dargs)
    @test fname == "animalz_rtime_pseudo_performance_distr_84772501_v7.hdf5"
    _dargs = EventOnsetDecoding.DecoderArgs(dargs;save_sample_indices=true)
    fname = EventOnsetDecoding.get_filename(_dargs)
    @test fname == "animalz_rtime_pseudo_performance_distr_1d926c0e_v7.hdf5"
    
    _fname = EventOnsetDecoding.get_filename(dargs, UInt32(1))
    @test _fname == "animalz_rtime_pseudo_performance_distr_3102fbf5_v7.hdf5"
    pr5, rr, f1score,fname = EventOnsetDecoding.run_rtime_decoder((counts=X, bins=bins,cellnames=cellnames), trialidx, 
                                                    tlabel, rtimes, dargs;decoder=MultivariateStats.MulticlassLDA,
                                                    redo=true, rseeds=rseeds, RNGType=StableRNG)
    
    @test f1score[:] ≈ fill(1.0, dargs.nruns)
    @test pr5[8,1,1,1,:] ≈ fill(1.0, dargs.nruns)
    @test pr5[1:7,1,1,1,:] ≈ fill(0.0, 7, dargs.nruns)

    # test precision
    bins = [-60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,10.0,15.0]
    nbins = length(bins)
    μ = 5.0*randn(rng, 1,ncells)
    X = 0.1*randn(rng, nbins,nt, ncells)

    X[12,:,:] .+= μ 
    X[13,:,:] .+= 0.75*μ
    X[14,:,:] .+= 0.5*μ
    dargs = EventOnsetDecoding.DecoderArgs(["animalZ/20140904/session01"],[1],1:ncells, windows=[5.0,10.0,15.0], latencies=[0.0, 5.0, 10.0], 
                                            difference_decoder=true,rtime_min=0.0, rtime_max=1.0,
                                            baseline_end=-20.0,nruns=20, mixin_postcue=true)
    rseeds = UInt32[0xf936675d, 0x1b56121c, 0xec4d628f, 0x88680ea1, 0xee7fcb1e, 0x33eaecb4,
                    0x93210719, 0x0e6769e1, 0x83679428, 0xa68cf7c7, 0x1a35f608, 0xf7ee9360,
                    0xc0d94b45, 0x334c08b5, 0x8a454eda, 0x4058d6b2, 0xabd023d9, 0x524c297d,
                    0xce218ede, 0x276fcc99]
    pr5, rr, f1score,fname = EventOnsetDecoding.run_rtime_decoder((counts=X, bins=bins,cellnames=cellnames), trialidx, 
                                                    tlabel, rtimes, dargs;decoder=MultivariateStats.MulticlassLDA,
                                                    redo=true, rseeds=rseeds, RNGType=StableRNG)

    # the training window contains the responses, so performance is maximum
    @test f1score[1,1,1,:] ≈ fill(1.0, dargs.nruns)
    μf = dropdims(mean(f1score,dims=4),dims=(3,4))
    @test μf[2,1] <= μf[1,1]
    @test μf[3,1] <= μf[2,1]

    @test μf[1,2] <= μf[1,1]
    @test μf[2,2] <= μf[2,1]
    @test μf[3,2] <= μf[2,2]

    @test μf[1,3] <= μf[1,1]
    @test μf[2,3] <= μf[2,1]
    @test μf[3,3] <= μf[3,1]
end