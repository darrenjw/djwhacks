camera {
    location <1.2, 1.2, 1.2>
    look_at  <.5, .5, .5>
}

light_source { <0, 2, 1> color <1,1,1> }

height_field {
    png "fBm2d.png"

    smooth
    pigment {
        gradient y
        color_map {
            [ 0 color <.5 .5 .5> ]
            [ 1 color <1 1 1> ]
        }
    }

    scale <1, 1, 1>
}

