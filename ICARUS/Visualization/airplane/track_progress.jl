using GLMakie
using FileIO

# Function to read points from the file
function read_points_from_file(file_name)

    WAKE_DATA = Vector{Vector{Vector{Float64}}}(undef, 0)
    open(file_name, "r") do file
        # Read the file. The format is as follows:
        # Each batch of data starts with NTime = n and then there are k lines of data
        # each containing 3 floats (x, y, z).
        # After that there is an empty line and the next batch starts.
        # The file ends with an empty line.

        # Read until EOF
        while !eof(file)
            # Read the first line
            line = readline(file)
            NTime = parse(Int, line[8:end])

            # Push an empty array to the WAKE_DATA array
            push!(WAKE_DATA, Vector{Vector{Float64}}(undef, 0))

            # Read the rest of the batch until the empty line
            while line != " "
                line = readline(file)
                # Add the data to an array.
                # The data is in the form of a string, so we need to split it.
                # The data is also in the form of a string, so we need to parse it.
                if line != " "
                    push!(WAKE_DATA[NTime], parse.(Float64, split(line)))
                end
            end
        end
    end
    return WAKE_DATA
end

# Create a callback function for the slider
function update_plot(value, ax, points)
    # Clear the previous plot
    ax.scene[1][:children] = []

    # Plot the data for the
    for i in 1:length(points[value])
        scatter!(ax, points[value][i][1], points[value][i][2], points[value][i][3], markersize=0.5, color=:blue)
    end

    # Set the axis title
    ax.title = "Wake formation $value"
end

function read_airplane_surface(directory)
    # Find all files in the directory that end with OUT.WG
    files = filter(x -> occursin(r"OUT.WG", x), readdir(directory))
    # Sort the files
    sort!(files)
    # Get the number of files
    n_files = length(files)

    SURFACE_DATA = Vector{Vector{Vector{Float64}}}(undef, 0)
    for i in 1:n_files
        file_name = joinpath(directory, files[i])
        push!(SURFACE_DATA, Vector{Vector{Float64}}(undef, 0))
        open(file_name) do file
            while !eof(file_name)
                line = readline(file_name)
                if line != " "
                    push!(SURFACE_DATA[NTime], parse.(Float64, split(line)))
                end
            end
        end
    end
    return SURFACE_DATA
end

function plot_airplane_surface(ax, points)
    for i in eachindex(points)
        scatter!(ax, points[i][1], points[i][2], points[i][3], markersize=0.5, color=:blue)
    end
end


# Main function
function main()
    directory = "/home/tryfonas/data/Uni/Software/hermes/Data/3D/e190_takeoff_3_hd/m4.0000_AoA"
    file_name = "$directorywake.dat"

    wake_points = read_points_from_file(file_name)
    surf_points = read_airplane_surface(directory)


    # Create a figure
    fig = GLMakie.Figure()
    display(fig)
    ax = Axis3(fig[1, 1])

    wake_xs = Vector{Observable{Vector{Float64}}}(undef, 0)
    wake_ys = Vector{Observable{Vector{Float64}}}(undef, 0)
    wake_zs = Vector{Observable{Vector{Float64}}}(undef, 0)


    # Set the axis labels
    ax.xlabel = "x"
    ax.ylabel = "y"
    ax.zlabel = "z"

    # Set the axis title
    ax.title = "Wake formation"

    # Create a slider
    slider = Slider(fig[2, 1], range=1:1:length(wake_points), startvalue=1)

    # Add the callback function to the slider
    on(slider.value) do value
        update_plot(value, ax, wake_points)
    end

    # Show the figure
    display(fig)
    readline()

end

main()
