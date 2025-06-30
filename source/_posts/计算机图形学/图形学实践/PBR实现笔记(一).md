---
title: PBR实现笔记(一)
date: 2025-06-05 18:07:57
tags: [OpenGL, PBR]
categories: 
- [计算机图形学, 图形学实践]
description: 根据LearnOpenGL、RTR Edition 4和GAMES101实现的基于物理的渲染
---

# PBR
 
PBR(Physical-Based Rendering， 基于物理的渲染)指的是一些在不同程度上都基于与现实世界的物理原理更相符的基本理论所构成的渲染技术的集合。

有两大好处：

1. 可以直接以物理参数为依据来编写表面材质，而不必依靠粗劣的修改与调整来让光照效果看上去正常
2. 不论光照条件如何，这些材质看上去都会是正确的，而在非PBR的渲染管线当中有些东西就不会那么真实了

判断一种PBR光照模型是否是基于物理的，必须满足以下三个条件：

1. 基于微平面(Microfacet)的表面模型。
2. 能量守恒。
3. 应用基于物理的BRDF。

## 微平面模型

![img](https://learnopengl-cn.github.io/img/07/01/microfacets_light_rays.png)

这项理论认为，达到微观尺度之后任何平面都可以用被称为微平面(Microfacets)的细小镜面来进行描绘。

由于这些微平面已经微小到无法逐像素地继续对其进行区分，因此我们假设一个粗糙度(Roughness)参数，然后用统计学的方法来估计微平面的粗糙程度。

## 能量守恒

光照射到物体，往往会有两种情况，一部分发生了反射，一部分发生了折射。

反射比较简单，就是镜面反射。

而折射则又会有三种情况：

1. 内部损耗
2. 从内部多次碰撞后又从表面折射出
3. 从物体的另一个表面折射出

但一般都直接认为折射光被全部吸收，所以又有：

```C++
float kS = calculateSpecularComponent(...); // 反射/镜面 部分
float kD = 1.0 - ks;                        // 折射/漫反射 部分
```

## BRDF

这里只是为了完整性，稍微提一下。

BRDF：双向反射分布函数。这个函数的意义是计算每一束光对不透明物体最终的反射光量。

后面，我认为在学习PBR的时候需要一个很重要的知识——辐射度量学，而这正是我不太熟悉的。所以先准备先学习一下几个重要的概念吧。

# 辐射度量学

## 辐射通量(Flux)

**单位时间**释放(emitted)、反射(emitted)、透射(transmitted)或接受(received)的**能量**。其实就是辐射度量学中的功率概念。

公式是：
$$
\Phi = \frac{dQ}{dt}
$$

## 立体角

立体角虽然不是辐射度量学特有的概念，但是是后面一个很有必要知道的知识点。

和弧度的计算方式有点类似：$\alpha = \frac{l}{r}$，弧长除以半径。

立体角的计算方式是：$\Omega=\frac{A}{r^2}$，球面上的**投影面积**与**半径的平方**之比。

在辐射度量学中常用到的是叫做单位立体角的概念，我的理解是一个球面的微分和半径的平方比。

其计算方式利用GAMES101课件中的图：

![image-20250605200520293](https://cdn.jsdelivr.net/gh/Dagaz84521/DagazBlogPicture@main/img/image-20250605200520293.png)



对于球面上参数坐标为$(\theta,\phi)$的点，其微分面积可以近似看作是一个小长方形：

- 对于$\theta$上的微分长度，就是其弧长$rd\theta$。
- 对于$\phi$上的微分长度，也是弧长，但是这个弧所在的圆半径是$r\sin\theta$。所以这个弧长是$r\sin\theta d\phi$

对于这两个，可以理解是经线($\phi$)和纬线($\theta$)，经线的长度都相同，所以是半径为$r$的圆，而纬线长度是受纬度($\theta$)影响的。 

所以也就能比较直观地感受到这个单位角公式其实是受纬度影响的。
$$
d\omega = \frac{dA}{r^2} = \sin \theta d\theta d\phi
$$

## 辐射强度(Intensity)

辐射(发光)强度是**单位立体角**由点光源发出的**功率**(power)。所以就是辐射通量对单位角求微分：
$$
I(\omega) = \frac{d\Phi}{d\omega}
$$

## 辐照度(irradiance)

辐照度是每**单位面积入射**到一个**表面上一点**的**辐射通量(功率)**，辐射通量对单位面积求微分：
$$
E=\frac{d\Phi}{dA}
$$


## 辐射(radiance)

是指一个表面在**每单位立体角、每单位投影面积**上所发射(emitted)、反射(reflected)、透射(transmitted)或接收(received)的**辐射通量(功率)**。

所以就可以理解是**单位投影面积**的**辐射强度**或者是**单位立体角**的**辐照度**：
$$
L(p,\omega) = \frac{d^2\Phi(p,\omega)}{d\omega dA \cos \theta}
$$

# 反射率方程

$$
L_o(p,\omega_o)=\int_\Omega f_r(p,\omega_i,\omega_o)L_i(p,\omega_i)n·\omega_id\omega_i
$$

基于物理的渲染所坚定遵循的是一种被称为反射率方程(The Reflectance Equation)的渲染方程的特化版本。

这看起来真的令人头大，我们需要一部分一部分地拆解。

1. BRDF(双向反射分布函数，这就是接下来要讲的了)
2. 入射光的辐射度
3. 一个入射光和法线的点乘

其中，我们需要着重介绍的是BRDF

# BRDF

BRDF，或者说双向反射分布函数，它接受入射（光）方向$ω_i$，出射（观察）方向$ω_o$，平面法线$n$以及一个用来表示微平面粗糙程度的参数$a$作为函数的输入参数。BRDF可以近似的求出每束光线对一个给定了材质属性的平面上最终反射出来的光线所作出的贡献程度。举例来说，如果一个平面拥有完全光滑的表面（比如镜面），那么对于所有的入射光线$ω_i$（除了一束以外）而言BRDF函数都会返回0.0 ，只有一束与出射光线$ω_o$拥有相同（被反射）角度的光线会得到1.0这个返回值。

实际上BRDF不是一个固定的公式，我们后续要使用的是其中一种称为Cook-Torrance的BRDF模型，其包含了漫反射和镜面反射两个部分：
$$
f_r= k_df_{lambert}+k_sf_{cook-torrance}
$$
其中$k_d$入射光线中**被折射**部分的能量所占的比率，$k_s$是**被反射**部分的比率。

## 漫反射

$f_{lambert}$是漫反射部分，称作Lambertian漫反射，用如下公式表示：
$$
f_{lambert}=\frac{c}{\pi}
$$
c是表面的颜色。

## 镜面反射

Cook-Torrance BRDF的镜面反射函数：
$$
f_{cook-torrance}=\frac{DFG}{4(\omega_o·n)(\omega_i·n)}
$$
分子部分是三个字母分别代表一种类型的函数，各个函数分别用来近似的计算出表面反射特性的一个特定部分：

- **法线分布函数**(Normal **D**istribution Function)：估算在受到表面粗糙度的影响下，朝向方向与半程向量一致的微平面的数量。这是用来估算微平面的主要函数。
- **几何函数**(**G**eometry Function)：描述了微平面自成阴影的属性。当一个平面相对比较粗糙的时候，平面表面上的微平面有可能挡住其他的微平面从而减少表面所反射的光线。
- **菲涅尔方程**(**F**resnel Rquation)：菲涅尔方程描述的是在不同的表面角下表面所反射的光线所占的比率。

接下来一一讲解这三个函数。

### 法线分布函数

其作用是从统计学上近似地表示了与某些（半程）向量$h$取向一致的微平面的比率。

也就是说，对于给定的半程向量$h$，如果微平面中有35%与向量$h$是取向一致的，则法线分布函数或者说NDF将会返回0.35。

很多种NDF都可以从统计学上来估算微平面的总体取向度，只要给定一些粗糙度的参数

后面，我们实践的时候将会采用Trowbridge-Reitz GGX这个模型：
$$
NDF_{GGXTR}(n,h,\alpha)=\frac{\alpha^2}{\pi ((n·h)^2(\alpha^2-1)+1)^2}
$$
其中$h$是用来与平面上微平面做比较用的半程向量，$\alpha$表示表面粗糙度。

对于同一个半程向量$h$，在不同粗糙度的结果。

![img](https://learnopengl-cn.github.io/img/07/01/ndf.png)

当粗糙度很低（也就是说表面很光滑）的时候，与半程向量取向一致的微平面会高度集中在一个很小的半径范围内。由于这种集中性，NDF最终会生成一个非常明亮的斑点。但是当表面比较粗糙的时候，微平面的取向方向会更加的随机。你将会发现与hh向量取向一致的微平面分布在一个大得多的半径范围内，但是同时较低的集中性也会让我们的最终效果显得更加灰暗。

写成一个函数应该是这样：

```glsl
float D_GGX_TR(vec3 N, vec3 H, float a)
{
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;

    return nom / denom;
}
```

### 几何函数

![img](https://learnopengl-cn.github.io/img/07/01/geometry_shadowing.png)

对于微平面，有可能会发生图中所示现象。几何函数从统计学上近似的求得了微平面间相互遮蔽的比率，这种相互遮蔽会损耗光线的能量

和NDF类似，几何函数采用一个材料的粗糙度参数作为输入参数，粗糙度较高的表面其微平面间相互遮蔽的概率就越高。

这里要使用的是Schlick-GGX，这是一种GGX与Schlick-Beckmann近似的结合体：
$$
G_{SchlickGGX}(n,v,k)=\frac{n·v}{(n·v)(1-k)+k}
$$
其中$k$是粗糙度$\alpha$的重映射：
$$
k_{direct}=\frac{(\alpha+1)^2}{8} \\
k_{IBL} = \frac{\alpha^2}{2}
$$
其根据针对直接光照还是针对IBL光照有所不同。

同时注意到上面这张图：

![img](https://learnopengl-cn.github.io/img/07/01/geometry_shadowing.png)

为了有效的估算几何部分，需要将观察方向（几何遮蔽(Geometry Obstruction)）和光线方向向量（几何阴影(Geometry Shadowing)）都考虑进去。

可以使用**史密斯法（Smith’s method）**来把两者都纳入其中：
$$
G(n,v,l,k)=G_{sub}(n,v,k)G_{sub}(n,l,k)
$$
几何函数的GLSL代码：

```GLSL
float GeometrySchlickGGX(float NdotV, float k)
{
    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float k)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}
```

### 菲涅尔方程

菲涅尔方程描述的是**被反射的光线**对比**光线被折射**的部分所占的**比率**，这个比率会随着我们**观察的角度**不同而不同。

从体验上来说：当垂直观察的时候，任何物体或者材质表面都有一个基础反射率(Base Reflectivity)，但是如果以一定的角度往平面上看的时候所有反光都会变得明显起来。

这里LearnOpenGL写的其实不是很直观，反正我看我的桌面是看不出什么东西来。

我这么说吧，不知道有没有这种体验：看一个较为清澈的湖面时，我们看近处的湖面，是可以透过湖水看到湖底的，而远处则更多是倒影和反光。这就是菲涅尔效应。而菲涅尔方程就是用来描述这一效应的。

也就是说，观察方向和平面的法线角度越接近90°，则反光就会越明显。

菲涅尔方程是一个很复杂的方程，一般都会使用近似的方式来求解，这里使用Fresnel-Schlick近似法求得近似解：
$$
F_{Schlick}=F_0 + (1-F_0)(1-(h·v))^5
$$
其中$F_0$表示平面的基础反射率，可以使用一个方式来为大多数电介质表面定义了一个近似的基础反射率。$F_0$取最常见的电解质表面的平均值，一般取0.04。

基于金属表面特性，我们要么使用电介质的基础反射率要么就使用F0F0来作为表面颜色。因为金属表面会吸收所有折射光线而没有漫反射，所以我们可以直接使用表面颜色纹理来作为它们的基础反射率。

所以可以这样实现：

```glsl
vec3 F0 = vec3(0.04);
F0      = mix(F0, surfaceColor.rgb, metalness);
```

然后Fresnel Schlick近似可以用代码表示为：

```C++
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
```

至此，我们可以获得一个Cook-Torrance反射率方程

## Cook-Torrance反射率方程

$$
L_o(p,\omega_o)=\int_\Omega(\frac{DFG}{4(\omega_o·n)(\omega_i·n)})L_i(p,\omega_i)n·\omega_i \mathrm{d}\omega_i
$$



# PBR材质

一般PBR材质都会有五种纹理：

- **反照率**(Albedo)：反照率(Albedo)纹理为每一个金属的纹素(Texel)（纹理像素）指定表面颜色或者基础反射率。和漫反射纹理类似
- **法线**(Normal)：法线贴图使我们可以逐片段的指定独特的法线，来为表面制造出起伏不平的假象。
- **金属度**(Metallic)：金属(Metallic)贴图逐个纹素的指定该纹素是不是金属质地的。
- **粗糙度**(Roughness)：粗糙度(Roughness)贴图可以以纹素为单位指定某个表面有多粗糙。
- **AO**(Ambient Occlusion)：环境光遮蔽(Ambient Occlusion)贴图或者说AO贴图为表面和周围潜在的几何图形指定了一个额外的阴影因子。

# 写在最后

其实这里好多内容我的理解还是停留在表面，十分可惜。

但我的想法是，能通过一次又一次的学习，最终能达到一个掌握的地步。

接下来一个路径应该是先去看浅墨的PBR白皮书，希望能对整个PBR技术的一个发展路径有所了解。

然后一篇我也不希望写太多内容，所以这里开始分个P，88。

