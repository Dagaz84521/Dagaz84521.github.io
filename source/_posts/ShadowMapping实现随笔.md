---
title: ShadowMapping实现随笔
date: 2025-05-27 18:55:39
tags: [OpenGL, ShadowMapping, 帧缓冲, 立方体贴图]
categories: 
- [计算机图形学, 图形学实践]
description: 根据LearnOpenGL还有浅墨翻译的RTR4实现的ShadowMapping
cover: https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250606141316291.png
---

# ShadowMapping

> 1978年，Williams [1888]提出了一种通用的、基于z-buffer的渲染器，它可以在任意物体上快速生成阴影。其核心想法是从光源的位置出发，使用z-buffer来渲染整个场景，然后再生成阴影效果。能够被光源“看见”的任何物体都会被照亮，光源“看不见”的物体则都处于阴影中。实际上在图像渲染的时候，我们最终只需要这个z-buffer即可，即我们只需要场景的深度信息；因此在这个特殊的场景渲染中，我们可以关闭光照、纹理等选项，也不用向颜色缓冲写入任何值。
>
> 在从光源视角渲染整个场景之后，z-buffer中的每个像素现在代表了最靠近光源的物体深度值。我们将这个z-buffer中的内容叫做阴影贴图（shadow map），有时候也会称为阴影深度图（shadow depth map）或者阴影缓冲区（shadow buffer）等。为了使用这个阴影贴图来生成阴影效果，我们会从相机的位置来对场景进行第二次渲染。在渲染每个图元的时候，对于该图元所覆盖的每个像素位置，我们都会将其与阴影贴图进行深度比较：如果着色点比阴影贴图中对应位置到光源的距离更远，则说明该点位于阴影中，否则该点不在阴影中。该算法是利用纹理映射实现的，如图7.10所示。阴影映射是一种十分流行的算法，因为它的计算成本相对来说是可预测的。创建阴影贴图的开销，与需要渲染的图元数量大致呈线性关系，并且访问时间是常量。在光源和物体不发生移动的场景中（例如一些计算机辅助设计应用CAD中），我们可以只生成一次阴影贴图，并在每一帧中进行重复使用。

其实思路很容易懂。

我们先思考为什么会产生阴影？

因为光线没办法到达这里，对吧。更准确点说，在到达这个点A之前，先被某个点B拦住了。也就是说这条光路上，点B离光源L更近。

OK，再来想，我们为什么看不到被挡住的事物A？因为从物体上回来的光线被另一个事物B挡住了。在这条光路上，事物B离我的眼睛更近。

OK，假如我们在光源向光线方向看，能看到点B，但看不到点A。

所以，一个场景中的某个点，是否被阴影覆盖，就要看在光源能不能看到这个点。那么如何判断一个点能不能被光源看见，就是看在这个点到光源的这条光路上，是否存在一个离光源更近的点。

也就是这个图所表达的意思：

![img](https://learnopengl-cn.github.io/img/05/03/01/shadow_mapping_theory_spaces.png)

眼睛观察到的点P并不能在光源处被“光源”所看见，所以点P处在阴影之中。

也就是说在CP这条光路上，点C离光源比点P离光源更近。换一个更符合图形学的说法就是，对于光源观察空间来说，点P的深度比点C要大。

我们可以通过把光源所能看见的点都记录下来，然后比较这个点的深度，就可以知道这个点能不能被光源看见。而记录光源所能看见的点，就是一个从光源观察空间的深度贴图，也就是阴影贴图(Shadow Map)。

对于一个定向光，只需要一张正交投影的贴图就行了。而对于一个点光源，则需要用到立方体贴图。

接下来的

# 定向光阴影贴图

如何获得这个阴影贴图呢？就是将摄像头放在光源位置，进行一次场景渲染，只不过这个场景我们不需要其颜色信息，只需要知道其深度信息。

## 光源观察空间的MVP矩阵

我们先做定向光（比如阳光）。由于定向光，所有的光线都是**平行**的。所以不会出现透视的情况，用正交投影就行。

所以投影矩阵就是：`glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);`

观察矩阵就是光源位置放置一个摄像机，其观察方向是光线方向。当然，这里我是直接使用了glm库的lookAt函数：`glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));`

这里可以稍微再复习一下这个函数需要三个vec3类型的向量：**摄像机的位置**， **摄像机看的位置**和**世界空间的上方向**。

模型矩阵就是正常的场景绘制就行了。

## 帧缓冲的设置

> 用于写入颜色值的颜色缓冲、用于写入深度信息的深度缓冲和允许我们根据一些条件丢弃特定片段的模板缓冲，这些缓冲结合起来叫做帧缓冲(Framebuffer)，它被储存在GPU内存中的某处。OpenGL允许我们定义我们自己的帧缓冲，也就是说我们能够定义我们自己的颜色缓冲，甚至是深度缓冲和模板缓冲。

这里再复习一下帧缓冲是怎么使用的。

首先，我们绘制在屏幕上的内容实际上用的是默认帧缓冲，一共有两块。当一块绘制完了通过`glfwSwapBuffers(window);`交换到显示的，然后再对换下来的进行绘制。

除此之外，我们也可以自己定义一些帧缓冲，来实现对他们的渲染，然后利用这些帧缓冲实现一些效果。我们自己定义的帧缓冲需要满足以下条件：

- 附加至少一个缓冲（颜色、深度或模板缓冲）。
- 至少有一个颜色附件(Attachment)。
- 所有的附件都必须是完整的（保留了内存）。
- 每个缓冲都应该有相同的样本数(sample)。

接下来，我们就需要一个帧缓冲，用于记录光源观察空间的深度信息，并且将结果存在一个纹理图像中，这个纹理图像就是shadow map.

```C++
//创建一个分辨率为1024 * 1024的2D纹理，提供给帧缓冲使用
const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT=1024;
unsigned int depthMap;
glGenTextures(1, &depthMap);
glBindTexture(GL_TEXTURE_2D, depthMap);
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
             SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); 
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

//创建一个帧缓冲对象，并将生成的深度纹理作为帧缓冲的深度缓冲
unsigned int depthMapFBO;
glGenFrameBuffers(1, &depthMapFBO);
glBindFrameBuffer(GL_FRAMEBUFFER, depthMapFBO);
//将depthMap这个纹理图像作为帧缓冲的深度附件，也就是深度就会绘制到这个2D纹理上。
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
//因为只需要用到深度，无需颜色，所以需要显式地告诉OpenGL不使用颜色进行渲染。所以要将读写都设置成GL_NONE来完成这件事
glDrawBuffer(GL_NONE);
glReadBuffer(GL_NONE);
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```

## 绘制阴影贴图

OK，经过上面的步骤，我们已经实现了阴影贴图和一个帧缓冲用于绘制这个阴影贴图了。

那么接下来，我们就要在我们的主渲染循环中完成阴影贴图的绘制了。

首先，我们需要两个着色器，一个是用于绘制阴影贴图的，另一个则是用于显示这张阴影贴图的。

这里我用`simpleDepthShader`这个着色器来实现绘制阴影贴图，使用`DebugDepthQuad`来将这个阴影贴图可视化。

首先，我们要通过`simpleDepthShader`绘制阴影贴图

首先是顶点着色器，通过我们之前在二.1中推出的`lightProjection` 和`lightView`就能得到光源空间的变化矩阵`lightSpaceMatrix`。

然后就能计算出光源空间的顶点位置。

```glsl
#version 330 core

layout(location = 0) in vec3 position;

uniform mat4 lightSpaceMatrix; // lightSpaceMatrix = lightProjection * lightView
uniform mat4 model;

void main()
{
    gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
}
```

而在后面片段着色器中，由于我们前面已经将depthMap作为深度附件了，所以直接设置成空就行了，深度测试的时候就会自动将深度值记录在depthMap中了。

所以在while循环中，我们需要的就是传入通过`lightProjection` 和`lightView`得到的`lightSpaceMatrix`。

还有一个点需要注意的是，要把帧缓冲换成`depthMapFBO`，而不是默认帧缓冲，同时还要记得修改视口至我们设置的纹理分辨率：

```C++
float near_plane = 1.0f, far_plane = 7.5f;
//定向光采用正交投影
glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane); 
//lookAt函数的三个参数分别是 摄像机位置、摄像机观察的位置和世界空间的上方向。
glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
glm::mat4 lightSpaceMatrix = lightProjection * lightView;
simpleDepthShader.use();
simpleDepthShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
//修改视口至我们设置的纹理分辨率
glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
//把缓冲换成depthMapFBO
glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
glClear(GL_DEPTH_BUFFER_BIT);
//绘制整个场景，Model矩阵在这里设置
renderScene(simpleDepthShader);
```

这个时候depthMap就是我们在光照空间的深度信息了。

我们接下来可以先用一个debug的shader，来看看我们的结果：

```glsl
// 顶点着色器
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoords;

out vec2 TexCoords;

void main()
{
    gl_Position = vec4(position, 1.0);
    // Pass the texture coordinates to the fragment shader
    TexCoords = texCoords;
}

// 片段着色器
#version 330 core
in vec2 TexCoords;

uniform sampler2D depthMap;
uniform float near_plane;
uniform float far_plane;

out vec4 FragColor;

// 这个函数是使用透视投影需要使用的
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}

void main()
{
    float depth = texture(depthMap, TexCoords).r;
    // Convert depth to a color value for visualization
    FragColor = vec4(vec3(depth), 1.0);
}
```

注意在循环中切换视口和默认帧缓冲：

```C++
glBindFramebuffer(GL_FRAMEBUFFER, 0);
glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

// 将深度贴图绑定到默认帧缓冲
DebugDepthQuad.use();
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, depthMap);
renderQuad();
```

这样，我们就能看到阴影贴图了：

![image-20250528223128871](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250528223128933.png)

## 渲染阴影

接下来，我们就需要一个新的shader来绘制整个场景，并判断是否绘制阴影了。

这里是顶点着色器，没什么需要更多补充的内容，就是正常渲染的mvp矩阵。

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out VS_OUT{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace; //在光源空间中片段位置
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    vs_out.Normal = mat3(transpose(inverse(model))) * aNormal; // Correct normal transformation
    vs_out.TexCoords = aTexCoords;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
}
```

然后就是重头戏，片段着色器了。

首先是正常的一个渲染，采用的是Blinn-Phong光照模型：

```glsl
#version 330 core
in VS_OUT{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
} fs_in;
out vec4 FragColor;
uniform sampler2D diffuseTexture;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    [...]
}

void main()
{
    vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightColor = vec3(1.0);
    // Ambient
    vec3 ambient = 0.15 * color;
    // Diffuse
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    // Specular
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor;    
    // 计算阴影
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace, normal, lightDir);       
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;    

    FragColor = vec4(lighting, 1.0f);
}
```

其中有一个函数`ShadowCalculation`，主要是用来计算是否存在阴影的，存在则返回1.0，否则返回0.0：

```glsl
float ShadowCalculation(vec4 fragPosLightSpace)
{
    //1.执行透视除法
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    //2.变换到[0,1]的范围
    projCoords = projCoords * 0.5 + 0.5;
    //3.取得最近点的深度(使用[0,1]范围下的fragPosLight当坐标)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    //4.取得当前片段在光源视角下的深度
    float currentDepth = projCoords.z;
    //5.检查当前片段是否在阴影中
    float shadow = currentDepth  > closestDepth  ? 1.0 : 0.0;
    return shadow;
}
```

我们一行一行解释：

1. 当我们在顶点着色器输出一个裁切空间顶点位置到`gl_Position`时，OpenGL自动进行一个透视除法，将裁切空间坐标的范围-w到w转为-1到1，这要将x、y、z元素除以向量的w元素来实现。也就是说，在深度贴图中，我们传入的`gl_Position`是经过透视除法的，但是我们自己从顶点着色器传入的`FragPosLightSpace`并未进行这一操作，所以需要先进行这一步，保证其z值是在[-1,1]的。
2. 上面的`projCoords`的xyz分量都是[-1,1]（下面会指出这对于远平面之类的点才成立），而为了和深度贴图的深度相比较，z分量需要变换到[0,1]；为了作为从深度贴图中采样的坐标，xy分量也需要变换到[0,1]。所以整个`projCoords`向量都需要变换到[0,1]范围。
3. `shadowMap`存储的就是深度，所以直接通过texture就可以获得。
4. `projCoords`就是当前片段在光源观察空间的位置，所以直接获取其z值就行。
5. 如果`currentDepth  > closestDepth`说明看不见，所以返回1.0，否则返回0.0。（由于精度的问题，不要使用==）

完成了这个，就需要修改一下渲染循环了：

```C++
glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
ShadowMapping.use();
ShadowMapping.setMat4("lightSpaceMatrix", lightSpaceMatrix);
ShadowMapping.setVec3("lightPos", lightPos);
ShadowMapping.setVec3("viewPos", camera.Position);
ShadowMapping.setInt("diffuseTexture", 0);
ShadowMapping.setInt("shadowMap", 1);

glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
glm::mat4 view = camera.GetViewMatrix();
ShadowMapping.setMat4("projection", projection);
ShadowMapping.setMat4("view", view);
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, woodTexture);
glActiveTexture(GL_TEXTURE1);
glBindTexture(GL_TEXTURE_2D, depthMap);
renderScene(ShadowMapping);
```

所以就能实现一个大概的效果：

<center>
    <img src = "https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250528225513991.png"/>
</center>

可以看到，效果非常好。（好在哪，我请问了）

现在我们的画面已经显现了阴影了，但是还是存在很多问题，其中最令人瞩目的还是这个自阴影，也就是画面中的条纹。

还有一些问题，比如说这个凭空出现的阴影：

<center>
    <img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250528225828663.png" />
</center>

还有分明的明暗分界：

<center>
    <img src="https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250528225922492.png" />
</center>

我们接下来的任务就是优化这些问题。

## 阴影的优化

### **阴影失真**

要解决问题，首先要知道为什么会产生这个问题。

这种**阴影失真**产生的原因是由于我们的shadow map并非连续的。

每个像素中存储了深度值，但是显示在我们屏幕上的多个相近fragment，可能同时使用阴影贴图中的同一个像素。

这些fragment计算出来的深度可能小于阴影贴图的采样值，也有可能大于。这就导致了阴影失真的发生：

![img](https://picx.zhimg.com/v2-7c4be5cc0842d46d6ba0a817ff157785_1440w.jpg)

图中abcd都使用了中间点的深度，cd显然更远。所以cd产生了阴影。

解决办法也很简单，只需要将这个cd两点的深度值减去一个bias就行了。

这个bias可以很小，0.005就能有不错的效果了。

但是有些表面坡度很大，仍然会产生阴影失真。有一个更加可靠的办法能够根据表面朝向光线的角度更改偏移量：使用点乘：

```glsl
float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
float shadow = currentDepth - bias  > closestDepth  ? 1.0 : 0.0;
```

增加了这个之后，失真立马就没了：

![image-20250529211132023](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250529211139240.png)

但是这个明暗分界还是在，而且还是会凭空产生阴影。

### 贴图环绕方式

凭空产生阴影，这个阴影其实和我们所想要的阴影是相同形状的。所以产生的原因也很简单，因为我们设置的超出深度贴图的方式是repeat的：

```C++
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); 
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
```

所以我们需要做的是就是超出了这个范围都修改成没有阴影就行了：

```C++
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
```

![image-20250529211847004](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250529211847200.png)

重复确实没有了。但这个黑白分界还是很明显。

### 黑白分界

这是因为那里的坐标超出了光的正交视锥的远平面。你可以看到这片黑色区域总是出现在光源视锥的极远处。

当一个点比光的远平面还要远时，它的投影坐标的z坐标大于1.0。这种情况下，GL_CLAMP_TO_BORDER环绕方式不起作用，因为我们把坐标的z元素和深度贴图的值进行了对比；它总是为大于1.0的z返回true。

所以也很简单，z大于1.0的时候，我们直接返回0.0就行了：

```glsl
float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
	[...]
    if(projCoords.z > 1.0)
        shadow = 0.0;

    return shadow;
}
```

![image-20250529212307364](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250529212307561.png)

这样大毛病就修复完了，接下来是小毛病了。

### Peter Pan

![image-20250529212427503](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250529212427692.png)

可以看到这个位置，阴影稍微有点偏移。

### PCF

我们可以看到，我们现在的阴影还是棱角分明的。

接下来要做的就是如何将这个阴影变得柔和。

有点类似于MSAA？就是在阴影周围多采样几个点，然后取一个平均值。

通过这种方法能让阴影边界有一种渐变的感觉，也就是周边的阴影值不再是1.0，而是一个处于0.0~1.0之间的浮点数：

```glsl
float shadow = 0.0;
vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
for(int x = -1; x <= 1; ++x)
{
    for(int y = -1; y <= 1; ++y)
    {
        float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
        shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
    }    
}
shadow /= 9.0;
```

进行一个3×3范围的一个超采样后，取平均值：
![image-20250604135403747](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250604135410966.png)

从远处看，确实有好一些，但是从近处看，似乎还是不那么美丽：

![image-20250604135538547](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/20250604135538769.png)

所以我又去查看了一下RTR4阴影那一节的部分，其中也讲到了PCF，也整理了一些前人得到的经验。也提出了更多更深一步的技术，但是，定向光的阴影贴图就先做到这，后面等RTR4整理到这一章后，我们再来实现后面的技术。这一部分的内容预期将会更新在附录的部分。

# 点光源阴影贴图

点光源和定向光还是有所不同的。

首先就是投影方式不同，对于点光源来说，其光线照在一个平面上的方式应该更像我们眼睛看事物，也就是使用了透视投影。

此外还有一个问题就是，点光源的方向是朝着四面八方的，不像定向光，有一个明确的前后。这就需要使用一个立方体贴图来存储这部分的信息了。

![img](https://learnopengl-cn.github.io/img/05/03/02/point_shadows_diagram.png)

所以我们要做的第一件事，就很简单咯，我们先需要创建一个立方体贴图，供我们的帧缓冲存储深度信息。

## 立方体贴图的创建

```C++
unsigned int depthCubemap;
glGenTextures(1, &depthCubemap);

const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);

//设置六个面的阴影贴图
for (unsigned int i = 0; i < 6; ++i)
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

//纹理参数
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
```

在正常情况下，需要立方体贴图纹理的一个面附加到帧缓冲对象上，渲染场景6次，每次将帧缓冲的深度缓冲目标改成不同立方体贴图面。但是通过几何着色器，我们就可以实现所有面在一个过程渲染。所以可以直接绑定这个立方体贴图`depthCubeMap`到我们的`depthMapFBO`帧缓冲上。

## 光空间的变换

与阴影映射教程类似，我们将需要一个光空间的变换矩阵T，但是这次是每个面都有一个。

对于一个面的P矩阵来说，还是很简单的：

```C++
float aspect = (GLfloat)SHADOW_WIDTH/(GLfloat)SHADOW_HEIGHT;
float near = 1.0f;
float far = 25.0f;
glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), aspect, near, far);
```

根据前面的分析，这里点光源需要用的是一个透视投影矩阵。

但是观察矩阵V就比较头疼了，因为每次看的都是不同方向的面，所以我们一共需要6个V矩阵。而且我们还需要按照右、左、上、下、近、远的顺序传入这个矩阵（因为我要在几何着色器完成六个面的深度绘制）：

```C++
std::vector<glm::mat4> shadowTransforms;
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0)));
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(-1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0)));
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(0.0,1.0,0.0), glm::vec3(0.0,0.0,1.0)));
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(0.0,-1.0,0.0), glm::vec3(0.0,0.0,-1.0)));
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,-1.0,0.0)));
shadowTransforms.push_back(shadowProj * 
                 glm::lookAt(lightPos, lightPos + glm::vec3(0.0,0.0,-1.0), glm::vec3(0.0,-1.0,0.0)));
```

但是这六个矩阵不是那么容易理解。

![缩略图](https://uploads.disquscdn.com/images/317909012d5718e941aab3aad496bbc5673be4f377327a20b3b9cae2db5fc943.png?w=800&h=367)

这张图是从[立方体贴图纹理 - OpenGL Wiki](https://www.khronos.org/opengl/wiki/Cubemap_Texture)获取的，蓝色的表示这个纹理面的上方向，也就是`lookAt`中最后一个变量的方向。（这是一种约定吗？不太懂，后序可以问一下。）

## 阴影贴图着色器

### 顶点着色器

顶点着色器做的工作比较简单：

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;

void main()
{
    gl_Position = model * vec4(aPos, 1.0);
}
```

### 几何着色器

几何着色器就是将我们从前面获取到的三角形，向六个面都做一个MVP变化：

```glsl
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 shadowMatrices[6];

out vec4 FragPos; 

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face; 
        for(int i = 0; i < 3; ++i) 
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
} 
```

其中gl_Layer是一个内建变量，它指定发散出基本图形送到立方体贴图的哪个面。

### 片段着色器

片段着色器所做的就是将深度信息保存下来：

```glsl
#version 330 core
in vec4 FragPos;

uniform vec3 lightPos;
uniform float far_plane;

void main()
{
    float lightDistance = length(FragPos.xyz - lightPos);
    
    lightDistance = lightDistance / far_plane;
    
    gl_FragDepth = lightDistance;
}
```

这里要进行归一化的原因是我们使用的深度缓存，其范围是[0.0,1.0]的浮点数。

以上，我们就能将场景的深度信息保存在一个立方体贴图中了。

接下来就需要使用这个立方体贴图完成阴影的绘制了。

## 渲染阴影

其他的部分其实和定向光差别不大。但是由于我们采用的是一个立方体内部的场景，而我们一般设置的立方体其法线都是面的法线，是朝立方体外部的，所以需要在片段着色器的部分增加一个bool类型的变量`reverse_normals`，来告诉我们渲染的立方体内部还是外部：

### 顶点着色器

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform bool reverse_normals;

void main()
{
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    if(reverse_normals) 
        vs_out.Normal = transpose(inverse(mat3(model))) * (-1.0 * aNormal);
    else
        vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
    vs_out.TexCoords = aTexCoords;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
```

### 片段着色器

```C++
#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} fs_in;

uniform sampler2D diffuseTexture;
uniform samplerCube depthMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform float far_plane;
uniform bool shadows;

float ShadowCalculation(vec3 fragPos)
{
    // 片段和光源向量
    vec3 fragToLight = fragPos - lightPos;
    // 利用fragToLight可以从立方体贴图中获取深度信息
    float closestDepth = texture(depthMap, fragToLight).r;
    // 将深度信息还原
    closestDepth *= far_plane;
    // 片段到光源的距离
    float currentDepth = length(fragToLight);
    
    float bias = 0.05; 
    float shadow = currentDepth -  bias > closestDepth ? 1.0 : 0.0;

    return shadow;
}

void main()
{           
    vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightColor = vec3(0.3);
    // ambient
    vec3 ambient = 0.3 * lightColor;
    // diffuse
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    // specular
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor;    
    // calculate shadow
    float shadow = shadows ? ShadowCalculation(fs_in.FragPos) : 0.0;                      
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;    
    
    FragColor = vec4(lighting, 1.0);
}
```

### 渲染循环

```C++
while(!glfwWindowShouldClose(window))
{
    // per-frame time logic
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    processInput(window);
    lightPos.z = static_cast<float>(sin(glfwGetTime() * 0.5) * 3.0);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 光源空间矩阵
    float near_plane = 1.0f, far_plane = 25.0f;
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), (float)SHADOW_WIDTH / (float)SHADOW_HEIGHT, near_plane, far_plane);
    std::vector<glm::mat4> shadowTransforms;
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f)));
    // 渲染阴影贴图
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        simpleDepth.use();
        for (unsigned int i = 0; i < 6; ++i)
            simpleDepth.setMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
        simpleDepth.setFloat("far_plane", far_plane);
        simpleDepth.setVec3("lightPos", lightPos);
        renderScene(simpleDepth);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pointShadow.use();
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = camera.GetViewMatrix();
    pointShadow.setMat4("projection", projection);
    pointShadow.setMat4("view", view);
    // set lighting uniforms
    pointShadow.setVec3("lightPos", lightPos);
    pointShadow.setVec3("viewPos", camera.Position);
    pointShadow.setInt("shadows", shadows); // enable/disable shadows by pressing 'SPACE'
    pointShadow.setFloat("far_plane", far_plane);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, woodTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubeMap);
    renderScene(pointShadow);
    // draw light source
    LightCube.use();
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, lightPos);
    model = glm::scale(model, glm::vec3(0.2f)); // a smaller cube
    LightCube.setMat4("model", model);
    LightCube.setMat4("projection", projection);
    LightCube.setMat4("view", view);
    glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubeMap);
    renderCube();


    glfwSwapBuffers(window);
    glfwPollEvents();
}
```

其中LightCube是为了能更直观地显示光源位置：

![image-20250605135255467](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250605135255467.png)

然后还可以像定向光一样，使用PCF优化。但是我觉得后面应该会使用不少优化技巧，所以先浅尝辄止吧。

# 附录

## 代码

