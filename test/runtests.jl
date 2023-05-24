using EventOnsetDecoding
using EventOnsetDecoding: MultivariateStats
using StableRNGs
using Test


@testset begin "Basic"
    # a highly simplistic model
    bins = [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,10.0,15.0]
    nbins = length(bins)
    nt = 100
    ncells = 5
    cellnames = ["animalZ/20140904/session01/array01/channel$(string(i,pad=3))/cell01" for i in 1:ncells]
    μ = 5.0*randn(1,ncells)
    X = 0.1*randn(nbins,nt, ncells)

    X[8,:,:] .+= μ 
    X[9,:,:] .+= 0.75*μ
    X[10,:,:] .+= 0.5*μ
    trialidx = [[1:nt;] for i in 1:ncells]
    tlabel = [fill(1,nt) for i in 1:ncells]
    rtimes = Dict("animalZ/20140904/session01"=>fill(0.1, nt)) 
    rseeds = UInt32[0xaa815070, 0xcb646653, 0x584bf898, 0xd099425a, 0x20e80af4]
    dargs = EventOnsetDecoding.DecoderArgs(["animalZ/20140904/session01"],[1],1:ncells, windows=[5.0], latencies=[0.0], 
                                            difference_decoder=true,rtime_min=0.0, rtime_max=1.0,
                                            baseline_end=-10.0,nruns=5, mixin_postcue=true)

    fname = EventOnsetDecoding.get_filename(dargs)
    @test fname == "animalz_rtime_pseudo_performance_distr_84772501_v7.hdf5"

    pr5, rr, f1score,fname = EventOnsetDecoding.run_rtime_decoder((counts=X, bins=bins,cellnames=cellnames), trialidx, 
                                                    tlabel, rtimes, dargs;decoder=MultivariateStats.MulticlassLDA,
                                                    redo=true, rseeds=rseeds, RNGType=StableRNG)
    
    @test f1score[:] ≈ fill(1.0, dargs.nruns)
    @test pr5[7,1,1,1,:] ≈ fill(1.0, dargs.nruns)
    @test pr5[1:6,1,1,1,:] ≈ fill(0.0, 6, dargs.nruns)

end