---
title: PBR实现笔记(三)
date: 2025-06-08 21:42:05
tags: [OpenGL, PBR]
categories: 
- [计算机图形学, 图形学实践]
description: 终于到了IBL了，这两节IBL结束，就可以展示一波了。
cover: https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250610110729003.png
---

OK啊，现在终于是要把IBL(Image based lighting，基于图像的光照)加入到我们的PBR中。

# IBL

> 基于图像的光照(Image based lighting, IBL)是一类光照技术的集合。其光源不是可分解的直接光源，而是将周围环境整体视为一个大光源。IBL 通常使用（取自现实世界或从3D场景生成的）环境立方体贴图 (Cubemap) ，我们可以将立方体贴图的每个像素视为光源，在渲染方程中直接使用它。这种方式可以有效地捕捉环境的全局光照和氛围，使物体**更好地融入**其环境。

$$
L_o(p,\omega)=\int_\Omega(k_d\frac{c}{\pi}+k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i d \omega_i
$$

这个公式，我们已经见过好几次了。在之前，我们使用了四个点光源将这个积分简化成了四个点光源的加和。

现在，引入了IBL后我们可以捕捉部分甚至全部的环境光照，这将会是一种更为精确的环境光照输入格式，也可以说是一种全局光照的粗略近似。这样就能看起来更加准确。

但随之而来的问题是，这次光源不再是之前的点光源，来自周围环境的每个方向$\mathcal{w}_i$的入射光都可能具有辐射度，这就有两个要求：

1. 给定任何方向向量$\mathcal{w}_i$ ，我们需要一些方法来获取这个方向上场景的辐射度
1. 解决积分需要快速且实时。

有一个思路来解决1，通过表示环境或场景辐照度的（预处理过的）环境立方体贴图，我们可以将立方体贴图的每个纹素视为一个光源。使用一个方向向量 $\mathcal{w}_i$对此立方体贴图进行采样，我们就可以获取该方向上的场景辐照度：

```glsl
vec3 radiance =  texture(_cubemapEnvironment, w_i).rgb;
```

为了以更有效的方式解决积分，我们需要对其大部分结果进行预处理——或称预计算。为此，我们必须深入研究反射方程：

## 反射方程分析

$$
L_o(p,\omega)=\int_\Omega(k_d\frac{c}{\pi}+k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i d \omega_i
$$

根据积分的性质，我们可以把这个反射方程改写成两部分：
$$
L_o(p,\omega)=\int_\Omega(k_d\frac{c}{\pi})L_i(p,\omega_i)n·\omega_i d \omega_i + 
\int_\Omega (k_s\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i d \omega_i
$$
在上一节讲这个公式的时候，我们知道加号左边的是漫反射部分，而加号右边的是镜面反射部分，也就是高光部分。

对于这两个积分，我们可以分开来求解。对于漫反射部分：
$$
L_{os}(p,\omega)=\int_\Omega(k_d\frac{c}{\pi})L_i(p,\omega_i)n·\omega_i d \omega_i
$$
而其中的$k_d\frac{c}{\pi}$，是一个常数，我们可以将其移动到积分外面：
$$
L_{os}(p,\omega)=(k_d\frac{c}{\pi})\int_\Omega L_i(p,\omega_i)n·\omega_i d \omega_i
$$
这给了我们一个只依赖于$\mathcal{w}_i$的积分（假设 $p$ 位于环境贴图的中心）。有了这些知识，我们就可以计算或预计算一个新的立方体贴图，它在每个采样方向——也就是纹素——中存储漫反射积分的结果，这些结果是通过卷积计算出来的。

为了对环境贴图进行卷积，我们通过对半球$\Omega$上的大量方向进行离散采样并对其辐射度取平均值，来计算每个输出采样方向$\mathcal{w}_o$的积分。用来采样方向$\mathcal{w}_i$的半球，要面向卷积的输出采样方向$\mathcal{w}_o$。

然后，将这个卷积结果存储到一个立方体贴图上，在每个采样方向$\mathcal{w}_o$上存储其积分结果，这样的立方体贴图被称为辐照度图。

# HDR环境贴图

谈及辐射度的文件格式，辐射度文件的格式（扩展名为 .hdr）存储了一张完整的立方体贴图，所有六个面数据都是浮点数，允许指定 0.0 到 1.0 范围之外的颜色值，以使光线具有正确的颜色强度。

这和我之前在天空盒中使用的立方体贴图还不太一样，立方体贴图属于低动态范围(Low Dynamic Range, LDR)。我们直接使用各个面的图像的颜色值，其范围介于 0.0 和 1.0 之间，计算过程也是照值处理。而hdr格式的所有六个面数据都是浮点数，允许指定 0.0 到 1.0 范围之外的颜色值，以使光线具有正确的颜色强度。

而且这种格式的贴图采用的并非是立方体贴图，而是一种叫做**等距柱状投影图**(Equirectangular Map)的方式。有一点确实需要说明：水平视角附近分辨率较高，而底部和顶部方向分辨率较低,在大多数情况下，这是一个不错的折衷方案，因为对于几乎所有渲染器来说，大部分有意义的光照和环境信息都在水平视角附近方向。

首先就是先要将这个贴图加载进来，还是使用`stb_image.h`这个头文件库就行了：

```C++
// in <utils/loadTexture.h>
unsigned int loadHDRTexture(char const * path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    float *data = stbi_loadf(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data); // note how we specify the texture's data value to be float

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "HDR Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}
```

当然也可以直接使用等距柱状投影图获取环境信息，但是这些操作还是显得相对昂贵，在这种情况下，直接采样立方体贴图的性能更高。

## 从等距柱状投影到立方体贴图

要将等距柱状投影图转换为立方体贴图，我们需要渲染一个（单位）立方体，并从内部将等距柱状图投影到立方体的每个面，并将立方体的六个面的图像构造成立方体贴图。此立方体的顶点着色器只是按原样渲染立方体，并将其局部坐标作为 3D 采样向量传递给片段着色器：

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 localPos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    localPos = aPos;  
    gl_Position =  projection * view * vec4(localPos, 1.0);
}
```

然后通过一些数学手段（这里不太明白，问了Deepseek，但是不太确定是否正确，将补充在附录部分），将等距柱状投影到立方体贴图。

```glsl
#version 330 core
out vec4 FragColor;
in vec3 localPos;

uniform sampler2D equirectangularMap;

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{       
    vec2 uv = SampleSphericalMap(normalize(localPos)); // make sure to normalize localPos
    vec3 color = texture(equirectangularMap, uv).rgb;

    FragColor = vec4(color, 1.0);
}
```

为了后续能够使用，这个立方体贴图应该绑定至帧缓冲，这个和之前的点光源的阴影映射部分有点类似：

### 创建帧缓冲

```C++
unsigned int captureFBO, captureRBO;
glGenFramebuffers(1, &captureFBO);
glGenRenderbuffers(1, &captureRBO);

glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);  
```

### 生成立方体贴图

```C++
unsigned int envCubemap;
glGenTextures(1, &envCubemap);
glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
for (unsigned int i = 0; i < 6; ++i)
{
    // note that we store each face with 16 bit floating point values
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 
                 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
}
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
```

### 设置投影和视图矩阵

```C++
glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
glm::mat4 captureViews[] =
{
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
};
```

### 将HDR渲染至立方体贴图

```C++
equirectangularToCubemapShader.use();
equirectangularToCubemapShader.setInt("equirectangularMap", 0);
equirectangularToCubemapShader.setMat4("projection", captureProjection);
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, hdrTexture);

glViewport(0, 0, 512, 512); // don't forget to configure the viewport to the capture dimensions.
glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
for (unsigned int i = 0; i < 6; ++i)
{
    equirectangularToCubemapShader.setMat4("view", captureViews[i]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderCube();
}
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```

通过上述代码，我们就能成功将等距柱状投影图转化为立方体贴图，这个贴图通过渲染至绑定到`captureFBO`帧缓冲的`equirectangularMap`上。

### 显示在天空盒中

这里我写了一个段代码，用于查看到这一步的效果：

```C++
while(!glfwWindowShouldClose(window))
{
    // per-frame time logic
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    processInput(window);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 将立方体贴图渲染至天空盒
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    glDepthFunc(GL_LEQUAL); // change depth function so depth test passes when values are equal to depth buffer's content
    skyboxShader.use();
    glm::mat4 view = glm::mat4(glm::mat3(camera.GetViewMatrix())); // remove translation from the view matrix
    skyboxShader.setMat4("view", view);
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    skyboxShader.setMat4("projection", projection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    renderCube(); // render the skybox as a cube
    glDepthFunc(GL_LESS); // reset depth function

    glfwSwapBuffers(window);
    glfwPollEvents();
}
```

其中`skybox`着色器的代码是：

```C++
//vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = vec3(aPos.x, -aPos.y, aPos.z);
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
//fragment shader
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{    
    vec3 color = texture(skybox, TexCoords).rgb;
    // Gamma校正
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}
```

后面增加了一个Gamma矫正，因为原本的HDR是物理空间的，需要进行Gamma矫正。

可以看一下一个对比图：

<table>
    <tr>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250608225540897.png"/>Gamma校正</center></td>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250608225525929.png" />无Gamma校正</center></td>
    </tr>
</table>

这就是一个大致的效果，是的，我们终于有GIF了。

![IBL立方体贴图](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/IBL%E7%AB%8B%E6%96%B9%E4%BD%93%E8%B4%B4%E5%9B%BE.gif)

# 立方体贴图的卷积

我们已经获取到了立方体贴图，下一步就是需要使用这个立方体贴图进行间接漫反射光的积分。

但是有一个很困难的问题，就是如果我们对每一个片段都进行间接漫反射光的积分的话，这个计算量将会很大。

就拿上一节的多球场景，每个球的每一个片段，我们都需要进行求他们的间接漫反射光积分，物体一旦多了，这是很恐怖的计算量了。

但是，与直接光不同的是，间接的漫反射光其实是可以预计算的。对于法向量方向相同的片段，其间接漫反射光的积分是相同的。

所以我们只需要计算出各个方向，并记录在一个立方体贴图，后续只需要在这个立方体贴图上取值就行了。

## 帧缓冲

还是可以使用之前的创建的帧缓冲，只不过现在我们要重新绘制另一个贴图`irrandianceMap`了。

## 生成立方体贴图

```C++
unsigned int irradianceMap;
glGenTextures(1, &irradianceMap);
glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
for(unsigned int i = 0; i < 6; i++)
{
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
}
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
```

## 计算卷积

由于前面已经将等距柱状投影图转换为立方体贴图，所以我们这里的卷积就可以直接从立方体贴图中获取环境信息了：

```glsl
#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform samplerCube environmentMap;

const float PI = 3.14159265359;
void main()
{
    vec3 normal = normalize(WorldPos);

    vec3 irradiance = vec3(0.0);

    //卷积操作
    
    FragColor = vec4(irradiance, 1.0);
}
```

这里可能会有一个比较疑惑的点（至少我一开始还是有点疑惑的），为什么直接将WorldPos直接标准化了就能获得法向量了呢？

![img](https://learnopengl-cn.github.io/img/07/03/01/ibl_hemisphere_sample_normal.png)

在生成辐照度立方体贴图时，我们渲染一个中心在原点的单位立方体。立方体表面的每个片段对应一个唯一的方向，通过标准化其位置向量 `normalize(WorldPos)` 获得。这个方向向量有两个关键作用：

1. **计算参考方向**：它定义了我们要计算辐照度的法线方向 N
2. **存储坐标**：它也是辐照度值在立方体贴图中存储的位置

在实际使用时，当我们有一个表面点具有法线向量 N，我们只需用这个 N 去采样立方体贴图，就能得到预先计算好的、该法线方向对应的辐照度值。

因此，这里的 `normalize(WorldPos)` 并不是在获取几何表面法线，而是在获取当前计算的'法线方向标识'

接下来就是如何进行卷积的计算了。

我们先来看公式：
$$
L_{os}(p,\omega)=k_d\frac{c}{\pi}\int_\Omega L_i(p,\omega_i)n·\omega_i d \omega_i
$$
其中，单位立体角在第一节介绍立体角的时候有在GAMES101的那张图上有过：
$$
d\omega = \frac{dA}{r^2}=\frac{r^2\sin\theta d\theta d\phi}{r^2}=\sin\theta d\theta d\phi
$$
而根据点乘我们还能知道$n·\omega_i$的结果是$\cos\theta$，所以这个公式就能再化简：
$$
L_{os}(p,\omega)=k_d\frac{c}{\pi}\int_\Omega L_i(p,\omega_i)\cos\theta \sin\theta d\theta d\phi
$$
而对于半球面的积分，可以拆解成：
$$
L_{o}(p,\omega)=k_d\frac{c}{\pi}
\int_{\phi=0}^{2\pi}
\int_{\theta=0}^{0.5\pi}
 L_i(p,\omega_i)\cos\theta \sin\theta d\theta d\phi
$$
这里可以不清楚的，可以去了解一下球坐标系。其实就是纬度和经度。

求黎曼和：
$$
L_{o}(p,\omega)=k_d\frac{c\pi}{n1n2}
\sum_{\phi=0}^{n1}
\sum_{\theta=0}^{n2}
 L_i(p,\omega_i)\cos\theta \sin\theta
$$
这样其实就能求了，对$\phi \in[0,2\pi]$和$\theta \in[0,0.5\pi]$之间采样，然后把这些采样点加在一起就可以了。

整个GLSL代码应该是：

```GLSL
#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform samplerCube environmentMap;

const float PI = 3.14159265359;
void main()
{
    vec3 normal = normalize(WorldPos);

    vec3 irradiance = vec3(0.0);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0;

    for(float phi = 0.0; phi < 2*PI; phi+=sampleDelta)
    {
        for(float theta = 0.0; theta <0.5*PI; theta += sampleDelta)
        {
            vec3 tangentSample = vec3(cos(theta) * sin(phi), sin(theta), cos(theta) * cos(phi));
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;

            irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples += 1.0;
        }
    }
    irradiance = PI * irradiance / nrSamples;
    FragColor = vec4(irradiance, 1.0);
}
```

同时，和之前一样，需要对六个面分别渲染场景：

```GLSL
irradianceShader.use();
irradianceShader.setInt("environmentMap", 0);
irradianceShader.setMat4("projection", captureProjection);
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

glViewport(0, 0, 32, 32); // don't forget to configure the viewport to the capture dimensions.
glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
for (unsigned int i = 0; i < 6; ++i)
{
    irradianceShader.setMat4("view", captureViews[i]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderCube();
}
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```

由于辐照度图对所有周围的辐射值取了平均值，因此它丢失了大部分高频细节，所以我们可以以较低的分辨率（32x32）存储。

## 显示在天空盒中

方式也很简单，只需要把之前的天空盒代码换成`irradianceMap`就OK了，效果是这样：

![image-20250609225242494](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250609225242494.png)

看着很模糊就对了。

# PBR 和间接辐照度光照

终于来到最后一步了。这个时候，我们只需要将原本的fbr片段着色器中增加环境光带来的部分，在之前，我们只是简单地采用了`vec3 ambient = vec3(0.03) * albedo * ao;`来模拟环境光照，而现在，我们终于可以用上真正的环境光照了：

```GLSL
vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0); // 镜面反射部分
vec3 kD = vec3(1.0) - kS; // 漫反射部分
kD *= 1.0 - metallic; // 根据金属度调整漫反射部分

vec3 irradiance = texture(irradianceMap, N).rgb; // 从环境贴图获取辐照度
vec3 diffuse = albedo * irradiance; // 漫反射部分

vec3 ambient = vec3(0.0);
ambient = (kD * diffuse) * ao; // 使用irradiance贴图计算环境光照

vec3 color = ambient + Lo; // 最终颜色
```

这样我们就能看到当前的结果了：

<table>
    <tr>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250610110729003.png"/>使用IBL</center></td>
        <td><center><img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250610110807351.png" />使用albedo</center></td>
    </tr>
</table>

# 附录

## 从等距柱状投影到立方体贴图的数学过程
