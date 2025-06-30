---
title: PBR实现笔记(二)
date: 2025-06-07 22:38:15
tags:
  - OpenGL
  - PBR
categories:
  - - 计算机图形学
    - 图形学实践
cover: https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250607235533437.png
---

# 回顾

先来回顾一下上一篇博客的内容，在一中，我们学习了PBR这一基于物理渲染的三个理论基础：

1. 基于微平面(Microfacet)的表面模型。
2. 能量守恒。
3. 应用基于物理的BRDF。

还补充了一部分的辐射度量学的知识，里面主要有几个物理量：

1. 辐射通量：光的功率。
2. 立体角：球面上的**投影面积**与**半径的平方**之比。
3. 辐射强度：单位立体角上的光源功率（辐射通量）。
4. 辐照度：单位面积上的光源功率（辐射通量）。
5. 辐射：一个表面在**每单位立体角、每单位投影面积**上所发射(emitted)、反射(reflected)、透射(transmitted)或接收(received)的**辐射通量(功率)**

也了解到了**反射率方程**，对于其中我们还了解到了一种**双向反射分布函数**模型——**Cook-Torrance BRDF模型**，其兼有漫反射和镜面反射两个部分。

其中漫反射部分在大多数场景下使用的是Lambertian漫反射，而镜面反射则引入了三种函数：

1. **法线分布函数**：估算在受到表面粗糙度的影响下，朝向方向与半程向量一致的微平面的数量。这是用来估算微平面的主要函数。
2. **几何函数**：描述了微平面自成阴影的属性。当一个平面相对比较粗糙的时候，平面表面上的微平面有可能挡住其他的微平面从而减少表面所反射的光线。
3. **菲涅尔方程**：菲涅尔方程描述的是在不同的表面角下表面所反射的光线所占的比率。

从而，最终我们获得了一个反射率方程模型：
$$
L_o(p,\omega)=\int_\Omega(k_d\frac{c}{\pi}+k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i d \omega_i
$$
虽然我们确实获得了这么一个反射率方程，但是我们并不知道究竟应该如何使用它。尤其是$L_i$的部分。

既然是刚刚入门，不妨先将这么一个模型简化那么亿点点。

# 亿点点的简化

如果呢，在空间中只有一个点光源，对于这个点光源（体积忽略不计），我们只要在一个以点光源为球心，半径为r的球面上的任意一个点去观察这个点光源，其辐射率应该是相同的。而在这个点的一个半球领域$\Omega$中，其他方向的辐射率，都为0。

或者简单点说，就是点光源照到空间中某一点的光线只有一条（很简单的理由：两点确定一线）。所以对于辐射率的一个微分，我们就可以简单地直接定义为点光源的性质，而免去这一部分的计算。

然后再增加上光线的衰减和由于入射角，我们就能知道打在这一点的辐射率：

```glsl
vec3  lightColor  = vec3(23.47, 21.31, 20.79); //设定的光源性质
vec3  wi          = normalize(lightPos - fragPos);
float cosTheta    = max(dot(N, Wi), 0.0);
float attenuation = calculateAttenuation(fragPos, lightPos);
float radiance    = lightColor * attenuation * cosTheta;
```

而且，利用这种没有面积的点光源，在积分的计算上也很方便。有几个光源，就计算几次，然后加和在一起就行了。

此外，对于其他类型的单点光源（定向光和聚光灯），也可以采用同样的简化。

# PBR表面模型

主要内容都在片段着色器上进行，顶点着色器就比较简单了：

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;

void main()
{
    TexCoords = aTexCoords;
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;   

    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}
```

我们先来写一部分处理PBR表面的代码：

在上一节的PBR材质部分，我们了解了：

1. **反照率**(Albedo)：反照率(Albedo)纹理为每一个金属的纹素(Texel)（纹理像素）指定表面颜色或者基础反射率。和漫反射纹理类似
2. **法线**(Normal)：法线贴图使我们可以逐片段的指定独特的法线，来为表面制造出起伏不平的假象。
3. **金属度**(Metallic)：金属(Metallic)贴图逐个纹素的指定该纹素是不是金属质地的。
4. **粗糙度**(Roughness)：粗糙度(Roughness)贴图可以以纹素为单位指定某个表面有多粗糙。
5. **AO**(Ambient Occlusion)：环境光遮蔽(Ambient Occlusion)贴图或者说AO贴图为表面和周围潜在的几何图形指定了一个额外的阴影因子。

这五种贴图，但是目前呢，我们先不用贴图。这些参数直接设定为固定的参数，normal则是直接由物体的顶点数组提供：

```glsl
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

uniform vec3 camPos;

uniform vec3  albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;
```

其中`albedo`上一节提到的比较少，主要就是表面颜色。`metallic`在上节提到了，我们需要其进行计算金属表面的基础反射率。`roughness`上节就提到的比较多了，在我们的BRDF中，那法线分布函数和几何函数都需要用到`roughness`取得一个近似的结果。`ao`也提到的很少，根据我目前的知识，只能先说，在后面的IBL中会比较有用吧。

此外，我们还需要在main中计算法向量`N`和观察方向`V`。

```glsl
void main()
{
    vec3 N = normalize(Normal); 
    vec3 V = normalize(camPos - WorldPos);
    [...]
}
```

# 直接光照

这里，我们采用四个点光源为整个场景提供光照。这个积分，就只需要分别计算这四个点光源的结果，然后加在一起就行了。

```glsl
vec3 Lo = vec3(0.0);
for(int i = 0; i < 4; ++i) 
{
    vec3 L = normalize(lightPositions[i] - WorldPos);
    vec3 H = normalize(V + L);

    float distance    = length(lightPositions[i] - WorldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance     = lightColors[i] * attenuation; 
    [...]  

```

所以接下来，我们就可以拿掉那个积分符号，直接计算后面的部分了：
$$
(k_d\frac{c}{\pi}+k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i
$$
经过前面的简化，$L_i(p,\omega_i)$直接就可以用光源的性质来表示，这里使用的是`lightColors[i]`。

我们就先来解决最头疼的DFG吧。

## DFG

在上一节，我就已经整理了DFG三个函数的代码

我就直接贴这部分的内容了。

对于菲涅尔函数，我们需要知道一个$F_0$，表示0°入射角的反射率，或者说是直接(垂直)观察表面时有多少光线会被反射。在PBR金属流中我们简单地认为大多数的绝缘体在$F_0$为0.04的时候看起来视觉上是正确的，对于金属表面我们根据反射率特别地指定$F_0$

```glsl
vec3 F0 = vec3(0.04); 
F0      = mix(F0, albedo, metallic);
vec3 F  = fresnelSchlick(max(dot(H, V), 0.0), F0);
```

```glsl
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
```

所以可以直接计算出NDF和G两个函数：

```glsl
float NDF = DistributionGGX(N, H, roughness);       
float G   = GeometrySmith(N, V, L, roughness);    
```

再根据BRDF的公式算出高光(specular)部分：

```glsl
vec3 nominator    = NDF * G * F;
float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
vec3 specular     = nominator / denominator;  
```

不赖。

## $k_s$和$k_d$

菲涅尔方程直接给出了$k_s$，而$k_d$就很容易就能获得了：

```glsl
vec3 kS = F;
vec3 kD = vec3(1.0) - kS;

kD *= 1.0 - metallic;   //金属不会折射，所以不会有漫反射，所以需要减去金属度，来保证这一性质
```

## 汇总

```glsl
float NdotL = max(dot(N, L), 0.0);        
Lo += (kD * albedo / PI + specular) * radiance * NdotL;
```

## 加上我们的环境光

```glsl
vec3 ambient = vec3(0.03) * albedo * ao;
vec3 color   = ambient + Lo;  
```

## 此时的片段着色器代码

```glsl
#version 330
out vec4 FragColor;

in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

uniform vec3 camPos;

// PBR参数
uniform vec3  albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

// lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main()
{
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - WorldPos);

    vec3 F0 = vec3(0.04); // 默认F0值
    F0 = mix(F0, albedo, metallic); // 根据金属度调整F0

    vec3 Lo = vec3(0.0);
    for(int i=0; i<4; i++)
    {
        vec3 L = normalize(lightPositions[i] - WorldPos); // 假设光源方向为(0, 1, 0)
        vec3 H = normalize(L + V);
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 1e-6; // 防止除零
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS; // 漫反射部分
        kD *= 1.0 - metallic; // 根据金属度调整漫反射部分

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; // 漫反射和镜面反射结合
    }
    vec3 ambient = vec3(0.03) * albedo * ao; // 环境光照
    vec3 color = ambient + Lo; // 最终颜色

    FragColor = vec4(color, 1.0); // 输出颜色
}
```

## 结果

![image-20250607235745255](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250607235745255.png)

## 加上Gamma修正和HDR

Gamma修正稍微学了一点，HDR基本完全没学。所以直接加上了代码，可以对比一下结果：

<table>
    <tr>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250607235745255.png"/>Gamma校正+HDR</center></td>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250607235533437.png" />无Gamma校正+HDR</center></td>
    </tr>
</table>

# 写在最后

其实，一开始还是觉得PBR非常的不一般，和传统的光照模型Phong和Blinn-Phong完全不同。

但是其实吧，从最低层的一个逻辑来说，目前采用的整个模型其实和Phong式光照模型没什么太大差别。依旧是考虑了三个部分的光（ambient、diffuse、specular），然后加在一起。只不过phong的光照并不如PBR的光照模型那么仔细？
