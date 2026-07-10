#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform PushConstants {
    mat4 viewProjection;
    float pointSize;
} pushData;

layout(location = 0) out vec3 pointColor;

void main() {
    gl_Position = pushData.viewProjection * vec4(inPosition, 1.0);
    gl_PointSize = pushData.pointSize;
    pointColor = inColor;
}
