---
title: PBR实现笔记(四)
date: 2025-06-10 15:54:48
tags: [OpenGL, PBR]
categories: 
- [计算机图形学, 图形学实践] 
cover: https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611152304153.png
---

哦耶，终于只剩下了环境的镜面反射部分了。

接下来我们就开始吧。

# 镜面反射IBL

在上一节中，完成了漫反射的IBL，利用的是通过卷积的方式预计算的辐照度图作为环境光照的漫反射部分。接下来，要完成镜面反射这一部分的内容了。上一节，我们将前面的反射方程拆分成了两个部分，而后一部分的方程：
$$
L_o(p,\omega_o)=\int_\Omega (k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i d \omega_i
$$
就是镜面反射的部分了。但这和上一节求的漫反射还不一样，不能直接通过漫反射的方式来解决。原因是镜面反射的积分结果不仅依赖$\omega_i$还依赖$\omega_o$

Epic Games采用的方法是分割求和近似法将预计算分成两个单独的部分求解，再将两部分组合起来得到后文给出的预计算结果。分割求和近似法将镜面反射积分拆成两个独立的积分：
$$
L_o(p,\omega_o)=\int_\Omega L_i(p,\omega_i)d\omega_i \ *
\int_\Omega f_r(p,\omega_i,\omega_o)n·\omega_i d\omega_i
$$
是不是对前一部分很眼熟？没错，这和之前的漫反射是差不多的，后面少了一个$n·\omega_i$。

对前面这部分，我们就可以采用之前的预计算的处理方式。

后面这部分，就暂且不提了，等我们做完前面这部分的积分，再来思考后面这部分的积分该如何解决。

# 预滤波HDR环境贴图

卷积的第一部分被称为预滤波环境贴图，它类似于辐照度图，是预先计算的环境卷积贴图，但这次考虑了粗糙度。因为随着粗糙度的增加，参与环境贴图卷积的采样向量会更分散，导致反射更模糊，所以对于卷积的每个粗糙度级别，我们将按顺序把模糊后的结果存储在预滤波贴图的 mipmap 中。

我们使用 Cook-Torrance BRDF 的法线分布函数(NDF)生成采样向量及其散射强度，该函数将法线和视角方向作为输入。由于我们在卷积环境贴图时事先不知道视角方向，因此 Epic Games 假设视角方向——也就是镜面反射方向——总是等于输出采样方向$ω_o$，以作进一步近似。

预滤波环境贴图的方法与我们对辐射度贴图求卷积的方法非常相似。对于卷积的每个粗糙度级别，我们将按顺序把模糊后的结果存储在预滤波贴图的 mipmap 中。 首先，我们需要生成一个新的立方体贴图来保存预过滤的环境贴图数据。为了确保为其 mip 级别分配足够的内存，一个简单方法是调用 glGenerateMipmap。

```C++
unsigned int prefilterMap;
glGenTextures(1, &prefilterMap);
glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
for (unsigned int i = 0; i < 6; ++i)
{
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
}
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
```

和之前的漫反射不同的是，镜面反射的结果是依赖粗糙度的：

![img](https://learnopengl-cn.github.io/img/07/03/02/ibl_specular_lobe.png)

反射光线可能比较松散，也可能比较紧密，但是一定会围绕着反射向量$r$，除非表面极度粗糙。

考虑到大多数光线最终会反射到一个基于半向量的镜面波瓣内，采样时以类似的方式选取采样向量是有意义的，因为大部分其余的向量都被浪费掉了，这个过程称为重要性采样。

这里可能会有点理解上的问题，LearnOpenGL可能的想法是，一束入射光的角度为$\omega_i$，它的反射光会分散在这个波瓣里。我们只要将这个波瓣里的入射光积分起来，那么出射角度为$\omega_i$的反射光，其结果就是这个积分了。

## 蒙特卡洛积分

又要开始补充数学知识了，但是说实话，这个我也不是很懂。所以只能大概讲一讲。

先打个比方吧：

> 假设您想要计算一个国家所有公民的平均身高。为了得到结果，你可以测量**每个**公民并对他们的身高求平均，这样会得到你需要的**确切**答案。但是，由于大多数国家人海茫茫，这个方法不现实：需要花费太多精力和时间。
>
> 另一种方法是选择一个小得多的**完全随机**（无偏）的人口子集，测量他们的身高并对结果求平均。可能只测量 100 人，虽然答案并非绝对精确，但会得到一个相对接近真相的答案，这个理论被称作大数定律。

而蒙特卡洛积分正是建立在这个基础上，采用相同的方法来求解积分。

不为所有可能的（理论上是无限的）样本值$x$求解积分，而是简单地从总体中随机挑选样本$N$生成采样值并求平均。随着$N$的增加，我们的结果会越来越接近积分的精确结果：
$$
O=\int_a^bf(x)dx=\frac{1}{N}\sum_{i=0}^{N-1}\frac{f(x)}{pdf(x)}
$$
我们在$a$到$b$上采样$N$个随机样本，将它们加在一起并除以样本总数来取平均。$pdf$表示概率密度函数，它的含义是特定样本在整个样本集上发生的概率。

说的简单点，蒙特卡洛积分其实提供给了我们一种方法，让我们绕开直接求取积分值，而是通过计算更容易获得的函数值，来近似间接获得积分值。

当然，现在只是粗略的了解了一点，后面感觉可以再开一个专栏，专门做计算机图形学的数学知识整理。哎，数学还是没学够啊看来。

其中有一个蒙特卡洛积分的知识点就是有偏和无偏。有偏的意思就是，比如范围是[0,1]的积分，但是我采样范围只有[0.25,1]。这就会导致积分更接近后者的值。无偏则是按照[0,1]进行积分。

但更准确的定义是有偏是最后计算出的积分不等于期望。

还有一个是重要性采样。还是一个问题：重要性采样干什么的？

**重要性采样是一种改进蒙特卡洛积分效率的技术。它通过定义一个精心设计的概率密度函数（PDF）`q(x)` 来进行采样，其核心思想是让 `q(x)` 的大小反映出函数 `f(x)` 对目标积分值的相对重要性（特别是 `|f(x)|` 较大的区域分配更高的采样概率）。然后，对从 `q(x)` 抽取的样本点 `x_i` 计算重要性权重 `f(x_i)/q(x_i)`，并将其进行简单平均得到积分估计。这种方法旨在尽可能降低估计量的方差，从而显著提升计算效率。**

回到我们这里，对于一个点来说，所有的镜面反射光都集中在这个波瓣上了。那么这个波瓣外的，我们就可以少采样一点，甚至可以直接不采样。

我也不太能说明这是否是一种有偏估计，因为这个范围确实小了。

> 我认为，这里有偏和无偏的差距在于。如果是pdf(x)=0是无偏，而直接不采样是有偏。因为概率等于0不一定是不可能事件，而直接不采样就是一个不可能事件。

但是好像没什么太大所谓，因为图形学有一句很重要的话，只要看起来是对的那就是对的。所以，只要能解释得通，而且效果没啥问题，那就是没问题。

> 听我解释：假如一个函数的定积分范围是[0,1]，但是其只有在[0.25,0.5]上是不等于0的。那么，$\int_0^1$和$\int_{0.25}^{0.5}$的结果是一样的。

## 低差异序列

蒙特卡洛积分的采样方式多种多样，其效果也随着具体任务的情况有所不同。

默认情况下，每次采样都是我们熟悉的完全（伪）随机，不过利用半随机序列的某些属性，我们可以生成虽然是随机样本但具有一些有趣性质的样本向量。

低差异序列就是其中一种，虽然是随机样本，但是样本分布会更加均匀。

当使用低差异序列生成蒙特卡洛样本向量时，该过程称为**拟蒙特卡洛积分**。拟蒙特卡洛方法具有更快的收敛速度，这使得它对于性能繁重的应用很有用。

所以我们采样的核心就是：

> 只在某些区域生成采样向量，该区域围绕微表面半向量，受粗糙度限制。通过将拟蒙特卡洛采样与低差异序列相结合，并使用重要性采样偏置样本向量的方法，我们可以获得很高的收敛速度。因为我们求解的速度更快，所以要达到足够的近似度，我们所需要的样本更少。

我们将使用的序列被称为 Hammersley 序列，该序列是把十进制数字的二进制表示镜像翻转到小数点右边而得。

我们可以直接在着色器中生成：

```glsl
float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  
```

## 镜面反射方向

之前一直没有提到一个问题——镜面反射方向。

这里Epic Games采用了一个简化：

```glsl
vec3 N = normalize(w_o);
vec3 R = N;
vec3 V = R;
```

其实就是把镜面反射方向等于等于输出采样方向$\omega_o$

虽然在掠射的时候效果不是很好，但是可以简化很多计算。

## GGX重要性采样

之前说过了，我们的采样会根据粗糙度，偏向微表面的半向量的宏观反射方向。

大致的采样过程就是：开始一个大循环，生成一个随机（低差异）序列值，用该序列值在切线空间中生成样本向量，将样本向量变换到世界空间并对场景的辐射度采样。不同之处在于，我们现在使用低差异序列值作为输入来生成采样向量：

```glsl

```

## 代码及详细解析

这里的代码比较多，而且还不太容易理解，我就先PO出来，然后一块一块的慢慢解析吧。

```glsl
#version 330 core
out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;
uniform float roughness;

const float PI = 3.14159265359;

//这几个函数前面都说过了，
float DistributionGGX(vec3 N, vec3 H, float roughness); //这个是第二节中写过的
//低差异序列
float RadicalInverse_VdC(uint bits);
vec2 Hammersley(uint i, uint N);
//采样函数，返回了一个采样向量
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness);

void main()
{       
    vec3 N = normalize(WorldPos);//法线向量
    // 在环境卷积中，假设视线方向与反射方向一致
    vec3 R = N;//反射向量
    vec3 V = R;//视线向量

    const uint SAMPLE_COUNT = 1024u;
    float totalWeight = 0.0;   
    vec3 prefilteredColor = vec3(0.0);     
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        //生成低差异序列采样点
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        //根据GGX分布生成半角向量
        vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
        //入射光方向
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);
		
        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
            //微表面分布函数
            float D   = DistributionGGX(N, H, roughness);
            
            //概率密度函数
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 
			
            //计算mipmap层级
            float resolution = 512.0; // resolution of source cubemap (per face)
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            //采样并叠加环境贴图
            prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
            totalWeight      += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight; //加权归一化

    FragColor = vec4(prefilteredColor, 1.0);
}  
```

在C++程序中捕获预过滤 mipmap 级别：

```C++
prefilterShader.use();
prefilterShader.setInt("environmentMap", 0);
prefilterShader.setMat4("projection", captureProjection);
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
unsigned int maxMipLevels = 5;
for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
{
    // reisze framebuffer according to mip-level size.
    unsigned int mipWidth  = 128 * std::pow(0.5, mip);
    unsigned int mipHeight = 128 * std::pow(0.5, mip);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
    glViewport(0, 0, mipWidth, mipHeight);

    float roughness = (float)mip / (float)(maxMipLevels - 1);
    prefilterShader.setFloat("roughness", roughness);
    for (unsigned int i = 0; i < 6; ++i)
    {
        prefilterShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderCube();
    }
}
glBindFramebuffer(GL_FRAMEBUFFER, 0);   
```

可以显示看看结果MipMap的结果：

<table>
    <tr>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611142004902.png"/>mipmap=0.0</center></td>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611142146053.png" />mipmap=1.2</center></td>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611142210140.png" />mipmap=2.4</center></td>
    </tr>
</table>

# 预计算 BRDF

我们来计算后半部分的积分：
$$
\int_\Omega f_r(p,\omega_i,\omega_o)n·\omega_i d\omega_i
$$
经过化简（具体过程参考[镜面IBL - LearnOpenGL CN](https://learnopengl-cn.github.io/07 PBR/03 IBL/02 Specular IBL/#brdf)），可以化简成：
$$
F_0\int_\Omega f(p,\omega_i,\omega_o)(1-(1-\omega_o·h)^5)n·\omega_i d\omega_i+
\int_\Omega f(p,\omega_i,\omega_o)(1-\omega_o·h)^5n·\omega_i d\omega_i
$$
注意，这里的$f$并不包含菲涅尔方程项$F$。

这样，我们又将一个积分拆成了两个积分。

前面那项有一个$F_0$系数的，我们称作菲涅耳响应的系数，后面那一项称为偏差。

和之前卷积环境贴图类似，我们可以对 BRDF 方程求卷积，其输入是 $n$ 和 $\omega_o$ 的夹角，以及粗糙度，并将卷积的结果存储在纹理中。（X轴 $n$ 和 $\omega_o$ 的夹角，Y轴材质表面粗糙度）

有了这么一张图，我们就能对于物体表面的某个点P，知道了其法向量$n$与摄像机和这个点P的方向($\omega_o$)的夹角，还有这个位置的粗糙度。我们就能通过这个纹理，获得其R通道（系数）和G通道（偏差）的值。这样就能直接计算出整体镜面反射的后半部分的积分值。

## 片段着色器代码

```glsl
#version 330 core
out vec2 FragColor;
in vec2 TexCoords;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    // note that we use a different k for IBL
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec2 IntegrateBRDF(float NdotV, float roughness)
{
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0; 

    vec3 N = vec3(0.0, 0.0, 1.0);
    
    const uint SAMPLE_COUNT = 1024u;
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // generates a sample vector that's biased towards the
        // preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}
// ----------------------------------------------------------------------------
void main() 
{
    vec2 integratedBRDF = IntegrateBRDF(TexCoords.x, TexCoords.y);
    FragColor = integratedBRDF;
}
```

其实可以看到，大部分的卷积思路和之前的预滤波HDR环境贴图是一样的。

都是对波瓣进行重要性采样，只不过这里的积分函数变了，使用的是$f_r$，因为默认$L_i$是1

还有一个就是`GeometrySchlickGGX`函数变了一点

以及在C++文件里，记得绑定一张贴图，用于存储这BRDF贴图：

```C++
// 计算BRDF贴图
unsigned int brdfLUTTexture;
glGenTextures(1, &brdfLUTTexture);
glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

// 渲染BRDF贴图
glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

glViewport(0, 0, 512, 512);
brdf.use();
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
renderQuad();

glBindFramebuffer(GL_FRAMEBUFFER, 0);  
```

大致的一个效果是：

![image-20250611145941785](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611145941785.png)

# 完成 IBL 反射

到此，所有的复杂计算都完成了，剩下的就只需要开始查表，然后输出就行了。

首先是我们刚获取的两个预计算贴图——环境贴图和BRDF的2D LUT贴图：

```glsl
uniform samplerCube prefilterMap;
uniform sampler2D   brdfLUT;  
```

要使用prefilterMap还得知道到底是层级的mipmap，所以：

```glsl
const float MAX_REFLECTION_LOD = 4.0;
vec3 prefilteredColor = textureLod(prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb;
```

而要使用BRDF的2D LUT贴图，我们需要材质粗糙度和视线-法线夹角：

```glsl
vec2 envBRDF  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
```

这样，我们就能获得最终的镜面反射结果了：

```
vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);
```

但这里LearnOpenGL使用了直接用间接光菲涅尔项$F$代替$F_0$。不太理解为什么

所以完整的结果就是：

```glsl
vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
vec3 kS = F; // 镜面反射部分
vec3 kD = vec3(1.0) - kS; // 漫反射部分
kD *= 1.0 - metallic; // 根据金属度调整漫反射部分

vec3 irradiance = texture(irradianceMap, N).rgb; // 从环境贴图获取辐照度
vec3 diffuse = albedo * irradiance; // 漫反射部分

const float MAX_REFLECTION_LOD = 4.0;
vec3 prefilteredColor = textureLod(prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb;    
vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;

vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

vec3 ambient = (kD * diffuse + specular) * ao;

vec3 color = ambient + Lo;

color = color / (color + vec3(1.0));
// gamma correct
color = pow(color, vec3(1.0/2.2)); 

FragColor = vec4(color, 1.0); // 输出颜色
```

![image-20250611152304153](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611152304153.png)

![image-20250611152326950](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250611152326950.png)

## 使用PBR材质试试看

![image-20250615144945951](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250615144945951.png)
