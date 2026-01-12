using Distributions: pdf, Gamma
using DSP: conv
using LinearAlgebra

function fmri_act(t, t_on, t_off, t_del; del_resp=6, del_undr=16, dis_resp=1, dis_undr=1, amp_ratio=6)
# function to create fMRI HRF-convolved activation waveform with HRF
# parameters based on defaults from SPM
# by David Frey
#
# inputs:
# t - time vector (s)
# t_on - duration of stimulus "on" period (s)
# t_off - duration of stimulus "off" period (s)
# t_del - stimulus onset delay (s)
#
# HRF parameters (optional):
# del_resp - response delay (s)
# del_undr - undershoot delay (s)
# dis_resp - response dispersion (s)
# dis_undr - undershoot dispersion (s)
# amp_ratio - response:undershoot amplitude ratio
#
# outputs:
# ref - reference activation waveform
#
    
    # get tr
    tr = t[2] - t[1];
    nt = length(t);
    
    # make hrf
    nk = ceil(Int, 32/tr)  # kernel length as integer
    resp = pdf.(Gamma(del_resp/dis_resp, dis_resp/tr), 0:nk)
    undr = pdf.(Gamma(del_undr/dis_undr, dis_undr/tr), 0:nk)
    hrf = amp_ratio*resp - undr;
    hrf = hrf/sum(abs.(hrf[:]));

    # create stimulus waveform
    stim = 1*(mod.(t .- t_del, t_on + t_off) .>= t_off).*(t .>= t_del);

    # convolve to create activation waveform
    ref = conv(hrf, stim);
    ref = ref[1:nt];

end

function fmri_tscore(GLM, x; df=size(GLM, 1) - size(GLM, 2))
# function to calculate fMRI t-score maps from given timeseries and design matrix
# by David Frey
#
# inputs:
# GLM - design matrix (nt x nc)
# x - image timeseries ((N) x nt)
# df - degrees of freedom (optional, default = nt - nc)
#
# outputs:
# tscore - voxel-wise activation t-scores ((N) x nc)
# beta - voxel-wise activation beta coefficients ((N) x nc)
#

    # get sizes
    nt = size(x, ndims(x)) # number of time points
    sz = size(selectdim(x, ndims(x), 1)) # size of each time point image
    nv = prod(sz) # total number of voxels
    nc = size(GLM, 2) # number of regressors

    # check sizes
    @assert nt == size(GLM, 1) "Number of time points in x ($(nt)) must match number of rows in GLM ($(size(GLM,1)))"

    # reshape timeseries (time x voxels)
    xvec = reshape(abs.(x), nv, nt)'

    # calculate beta coefficients
    beta = pinv(GLM) * xvec

    # calculate variance of residuals
    r = xvec .- GLM * beta
    rss = ones(nt)' * (r .^ 2)
    v = rss / df

    # calculate t-scores
    tscore = zeros(nc, nv)
    GLM_gram = GLM' * GLM
    for i in 1:N^2
        SE_beta = sqrt.(v[i] * diag(pinv(GLM_gram)))
        tscore[:, i] = beta[:, i] ./ SE_beta[:]
    end

    # reshape outputs
    tscore = reshape(tscore', sz..., nc)
    beta = reshape(beta', sz..., nc)

    return tscore, beta

end