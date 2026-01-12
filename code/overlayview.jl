using ColorTypes: Colorant
using ColorSchemes
using Plots
using MIRTjim

# overlayview layer object
struct ovLayer
    img::AbstractMatrix{<:Real}
    clim::Tuple{<:Real, <:Real}
    color
    name::String
end

# layer constructor function
function ovLayer(img::AbstractMatrix{<:Real}, clim::Tuple{<:Real, <:Real}; color=:grays, name="")
    ovLayer(img,clim,color,name)
end

# function to show the layers
function ovshow(layers::Vector{ovLayer})
    
    # get the size of the first layer
    sz = size(layers[1].img);
    img_all = zeros(sz);
    color_all = Colorant[];

    for i in 1:length(layers)

        # check if layer is valid in size
        l_sz = size(layers[i].img);
        if l_sz != sz
            error("All layers must have the same size. Layer $i has size $(l_sz), expected $(sz).")
        end

        # normalize and center the layer image
        l_img = Float64.(layers[i].img);
        l_img .-= layers[i].clim[1];
        l_img ./= layers[i].clim[2] - layers[i].clim[1];
        l_img[l_img .> 1] .= 1.0;
        l_img[l_img .< 0] .= 0.0;

        # add to total image
        img_all[l_img .> 0] .= (i-1) .+ 255/256*l_img[l_img .> 0];

        # get the layer ColorSchemes
        cs = getfield(ColorSchemes, layers[i].color)
        append!(color_all, get(cs, range(0, 1, length=256)))

    end

    # return combined layers
    return Dict("img_all" => img_all, "color_all" => color_all, "clim_all" => (0, length(layers)))

end