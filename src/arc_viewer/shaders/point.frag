#version 450

layout(location = 0) in vec3 pointColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 centered = gl_PointCoord * 2.0 - 1.0;
    if (dot(centered, centered) > 1.0) {
        discard;
    }
    outColor = vec4(pointColor, 1.0);
}
