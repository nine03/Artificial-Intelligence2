# 卷积神经网络(Convolutional Neural Networks, CNN 컨볼루션 신경망)

### 主要原理(주요원리)
首先，先获取一个感性认识，下图是一个卷积神经网络的示意图：

![캡처](https://user-images.githubusercontent.com/60682087/89252295-4a473900-d654-11ea-9b79-1210230cf9d7.JPG)

### 网络架构

如图1所示，一个卷积神经网络由若干卷积层、Pooling层、全连接层组成。你可以构建各种不同的卷积神经网络，它的常用架构模式为：
INPUT -> [[CONV]*N -> POOL?]*M -> [FC]*K

也就是N个卷积层叠加，然后(可选)叠加一个Pooling层，重复这个结构M次，最后叠加K个全连接层。

对于图1展示的卷积神经网络：

INPUT -> CONV -> POOL -> CONV -> POOL -> FC -> FC

也就是：N=1, M=2, K=2。

### 三维的层结构

从图1可以发现卷积神经网络的层结构和全连接神经网络的层结构有很大不同。全连接神经网络每层的神经元是按照一维排列的，也就是排成一条线的样子；而卷积神经网络每层的神经元是按照三维排列的，也就是排成一个长方体的样子，有宽度、高度和深度。

对于图1展示的神经网络，我们看到输入层的宽度和高度对应于输入图像的宽度和高度，而它的深度为1。接着，第一个卷积层对这幅图像进行了卷积操作(后面我们会讲如何计算卷积)，得到了三个Feature Map。这里的"3"可能是让很多初学者迷惑的地方，实际上，就是这个卷积层包含三个Filter，也就是三套参数，每个Filter都可以把原始输入图像卷积得到一个Feature Map，三个Filter就可以得到三个Feature Map。至于一个卷积层可以有多少个Filter，那是可以自由设定的。也就是说，卷积层的Filter个数也是一个超参数。我们可以把Feature Map可以看做是通过卷积变换提取到的图像特征，三个Filter就对原始图像提取出三组不同的特征，也就是得到了三个Feature Map，也称做三个通道(channel)。

继续观察图1，在第一个卷积层之后，Pooling层对三个Feature Map做了下采样(后面我们会讲如何计算下采样)，得到了三个更小的Feature Map。接着，是第二个卷积层，它有5个Filter。每个Fitler都把前面下采样之后的3个**Feature Map卷积在一起，得到一个新的Feature Map。这样，5个Filter就得到了5个Feature Map。接着，是第二个Pooling，继续对5个Feature Map进行下采样**，得到了5个更小的Feature Map。

图1所示网络的最后两层是全连接层。第一个全连接层的每个神经元，和上一层5个Feature Map中的每个神经元相连，第二个全连接层(也就是输出层)的每个神经元，则和第一个全连接层的每个神经元相连，这样得到了整个网络的输出。

至此，我们对卷积神经网络有了最基本的感性认识。接下来，我们将介绍卷积神经网络中各种层的计算和训练。

### 卷积神经网络输出值的计算

我用一个简单的例子来讲述如何计算卷积，然后，我们抽象出卷积层的一些重要概念和计算方法。

假设有一个5*5的图像，使用一个3*3的filter进行卷积，想得到一个3*3的Feature Map，如下所示：

![캡처1](https://user-images.githubusercontent.com/60682087/89252921-1705a980-d656-11ea-9e00-64a9bbb8405c.JPG)

为了清楚的描述卷积计算过程，我们首先对图像的每个像素进行编号，用xi,xj表示图像的第i行第j列元素；对filter的每个权重进行编号，用wm,wn表示第m行第n列权重，用wb表示filter的偏置项；对Feature Map的每个元素进行编号，用ai,aj表示Feature Map的第i行第j列元素；用f表示激活函数(这个例子选择relu函数作为激活函数)。然后，使用下列公式计算卷积：

![캡처2](https://user-images.githubusercontent.com/60682087/89253105-9a26ff80-d656-11ea-9f17-fb8e9b3be519.JPG)

对于Feature Map左上角元素a0,0来说，其卷积计算方法为：

![캡처3](https://user-images.githubusercontent.com/60682087/89253157-c5115380-d656-11ea-8b37-71983651748c.JPG)

计算结果如下图所示：

![캡처4](https://user-images.githubusercontent.com/60682087/89253203-ea9e5d00-d656-11ea-9bcb-469e7b0edce7.JPG)

接下来，Feature Map的元素a0,1的卷积计算方法为：

![캡처5](https://user-images.githubusercontent.com/60682087/89253254-13beed80-d657-11ea-95c0-9971bd436596.JPG)

计算结果如下图所示：

![캡처6](https://user-images.githubusercontent.com/60682087/89253312-3ea94180-d657-11ea-8136-723f78fdeb2f.JPG)

可以依次计算出Feature Map中所有元素的值。下面的动画显示了整个Feature Map的计算过程：

![캡처7](https://user-images.githubusercontent.com/60682087/89253369-69939580-d657-11ea-8ed0-d30ae6929b3b.JPG)

![캡처8](https://user-images.githubusercontent.com/60682087/89253370-6a2c2c00-d657-11ea-8fa7-b244b1bcf90e.JPG)

上面的计算过程中，步幅(stride)为1。步幅可以设为大于1的数。例如，当步幅为2时，Feature Map计算如下：

![캡처9](https://user-images.githubusercontent.com/60682087/89253430-a069ab80-d657-11ea-8d62-ee240b95b9df.JPG)

![캡처10](https://user-images.githubusercontent.com/60682087/89253432-a1024200-d657-11ea-9f5e-7df848ebc0e5.JPG)

![캡처11](https://user-images.githubusercontent.com/60682087/89253434-a2336f00-d657-11ea-8427-fb0b07c9a443.JPG)

![캡처12](https://user-images.githubusercontent.com/60682087/89253438-a3649c00-d657-11ea-8d96-0a0ff0f8826f.JPG)

我注意到，当步幅设置为2的时候，Feature Map就变成2*2了。这说明图像大小、步幅和卷积后的Feature Map大小是有关系的。事实上，它们满足下面的关系：

![캡처13](https://user-images.githubusercontent.com/60682087/89253543-e32b8380-d657-11ea-9264-6ebca86859f3.JPG)

在上面两个公式中，W2是卷积后Feature Map的宽度；W1是卷积前图像的宽度；F是filter的宽度；P是Zero Padding数量，Zero Padding是指在原始图像周围补几圈0，如果P的值是1，那么就补1圈0；S是步幅；H2是卷积后Feature Map的高度；H1是卷积前图像的宽度。式2和式3本质上是一样的。

以前面的例子来说，图像宽度W1 = 5,filter宽度 F = 3，Zero Padding P = 0，步幅 S = 2, 则

![캡처14](https://user-images.githubusercontent.com/60682087/89253726-64831600-d658-11ea-8807-2ab684ff6008.JPG)

说明Feature Map宽度是2。同样，我们也可以计算出Feature Map高度也是2。

卷积前的图像深度为D，那么相应的filter的深度也必须为D。我们扩展一下式1，得到了深度大于1的卷积计算公式：

![캡처15](https://user-images.githubusercontent.com/60682087/89253800-96947800-d658-11ea-8ce7-89c6c5802e35.JPG)

在式4中，D是深度；F是filter的大小(宽度或高度，两者相同)；Wd,Wm,Wn表示filter的第d层第m行第n列权重；ad,ai,aj表示图像的第d层第i行第j列像素；其它的符号含义和式1是相同的，不再赘述。

每个卷积层可以有多个filter。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的。

下面的动画显示了包含两个filter的卷积层的计算。我们可以看到7*7*3输入，经过两个3*3*3filter的卷积(步幅为2)，得到了3*3*2的输出。另外我们也会看到下图的Zero padding是1，也就是在输入元素的周围补了一圈0。Zero padding对于图像边缘部分的特征提取是很有帮助的。

![캡처16](https://user-images.githubusercontent.com/60682087/89253962-fd199600-d658-11ea-9005-6b6af5f15cda.JPG)

以上就是卷积层的计算方法。这里面体现了局部连接和权值共享：每层神经元只和上一层部分神经元相连(卷积计算规则)，且filter的权值对于上一层所有神经元都是一样的。对于包含两个3*3*3的fitler的卷积层来说，其参数数量仅有(3*3*3+1)*2=56个，且参数数量与上一层神经元个数无关。与全连接神经网络相比，其参数数量大大减少了。

式4的表达很是繁冗，最好能简化一下。就像利用矩阵可以简化表达全连接神经网络的计算一样，我利用卷积公式可以简化卷积神经网络的表达。

### 二维卷积公式

设矩阵A,B，其行、列数分别为ma,na,mb,nb，则二维卷积公式如下：

![캡처17](https://user-images.githubusercontent.com/60682087/89287011-21439a00-d68e-11ea-87fa-acc7f1496e60.JPG)

且s,t满足条件0<= s < ma + mb -1 , 0 <= t < na + nb -1

可以把上式写成   C = A * B   (式5)

![캡처18](https://user-images.githubusercontent.com/60682087/89287228-87c8b800-d68e-11ea-958e-c1381620b209.JPG)

从上图可以看到，A左上角的值a0,0与B对应区块中右下角的值b1,1相乘，而不是与左上角的b0,0相乘。因此，数学中的卷积和卷积神经网络中的『卷积』还是有区别的，为了避免混淆，我们把卷积神经网络中的『卷积』操作叫做互相关(cross-correlation)操作。

卷积和互相关操作是可以转化的。首先，我把矩阵A翻转180度，然后再交换A和B的位置（即把B放在左边而把A放在右边。卷积满足交换率，这个操作不会导致结果变化），那么卷积就变成了互相关。

可以把式5代入到式4:

![캡처19](https://user-images.githubusercontent.com/60682087/89287775-82b83880-d68f-11ea-811b-c33b119635cf.JPG)

其中，A是卷积层输出的feature map。同式4相比，式6就简单多了。然而，这种简洁写法只适合步长为1的情况。

### Pooling层输出值的计算 

Pooling层主要的作用是下采样，通过去掉Feature Map中不重要的样本，进一步减少参数数量。Pooling的方法很多，最常用的是Max Pooling。Max Pooling实际上就是在n*n的样本中取最大值，作为采样后的样本值。下图是2*2 max pooling：

![캡처20](https://user-images.githubusercontent.com/60682087/89288039-f5c1af00-d68f-11ea-952c-c7607343294c.JPG)

除了Max Pooing之外，常用的还有Mean Pooling——取各样本的平均值。

对于深度为D的Feature Map，各层独立做Pooling，因此Pooling后的深度仍然为D。

### 卷积神经网络的训练

和全连接神经网络相比，卷积神经网络的训练要复杂一些。但训练的原理是一样的：利用链式求导计算损失函数对每个权重的偏导数（梯度），然后根据梯度下降公式更新权重。训练算法依然是反向传播算法。

1. 前向计算每个神经元的输出值aj(j表示网络的第j个神经元，以下同);
2. 反向计算每个神经元的误差项dj,dj在有的文献中也叫做敏感度(sensitivity)。它实际上是网络的损失函数Ed对神经元加权输入netj的偏导数，即
![캡처21](https://user-images.githubusercontent.com/60682087/89289094-a4b2ba80-d691-11ea-8eda-5f1ec1b1c4ef.JPG)
3. 计算每个神经元连接权重wji的梯度（wji表示从神经元i连接到神经元j的权重），公式为 ![캡처22](https://user-images.githubusercontent.com/60682087/89289274-ea6f8300-d691-11ea-98da-86a9b96a50ca.JPG) , 其中 , ai表示神经元i的输出。

最后，根据梯度下降法则更新每个权重即可。
对于卷积神经网络，由于涉及到局部连接、下采样的等操作，影响到了第二步误差项的具体计算方法，而权值共享影响了第三步权重的梯度的计算方法。接下来，我们分别介绍卷积层和Pooling层的训练算法。

### 卷积层的训练
对于卷积层，我们先来看看上面的第二步，即如何将误差项d传递到上一层；然后再来看看第三步，即如何计算filter每个权值w的梯度。
卷积层误差项的传递,最简单情况下误差项的传递。
先来考虑步长为1、输入的深度为1、filter个数为1的最简单的情况。
假设输入的大小为3*3，filter大小为2*2，按步长为1卷积，我们将得到2*2的feature map。如下图所示：

![캡처23](https://user-images.githubusercontent.com/60682087/89289932-04f62c00-d693-11ea-9782-c9b8bf17f6d7.JPG)

在上图中，为了描述方便，我为每个元素都进行了编号。用![캡처24](https://user-images.githubusercontent.com/60682087/89290343-b4330300-d693-11ea-97b6-76f0dff8ab7e.JPG) 表示第j-1 层第j行第j列的误差项；用wm,wn表示
filter第m行第n列权重，用wb表示filter的偏置项；用![캡처25](https://user-images.githubusercontent.com/60682087/89290632-1d1a7b00-d694-11ea-957c-80596b2e3760.JPG) 表示第j-1层第i行第j列神经元的输出；用![캡처26](https://user-images.githubusercontent.com/60682087/89290753-4cc98300-d694-11ea-89d2-9882f21beefc.JPG)
表示第j-1行神经元的加权输入；用![캡처27](https://user-images.githubusercontent.com/60682087/89290860-7682aa00-d694-11ea-8386-543a5a85fbb5.JPG)
表示第i层第j行第j列的误差项；用f^j-1表示第j-1层的激活函数。它们之间的关系如下：

![캡처28](https://user-images.githubusercontent.com/60682087/89291142-f0b32e80-d694-11ea-86e5-5eed8e1c5792.JPG)

先求第一项 ![캡처29](https://user-images.githubusercontent.com/60682087/89291186-0a547600-d695-11ea-8287-4743253630cc.JPG)。
先来看几个特例，然后从中总结出一般性的规律。

![캡처30](https://user-images.githubusercontent.com/60682087/89291346-57384c80-d695-11ea-8dc4-7e67ce034349.JPG)

![캡처31](https://user-images.githubusercontent.com/60682087/89291353-599aa680-d695-11ea-87c5-62d1d63af2fd.JPG)

![캡처32](https://user-images.githubusercontent.com/60682087/89291355-5acbd380-d695-11ea-8ea5-cb0c3b9dd4bc.JPG)

从上面三个例子，发挥一下想象力，不难发现，计算![캡처33](https://user-images.githubusercontent.com/60682087/89291461-8484fa80-d695-11ea-99ad-3ab6ca990e94.JPG)
，相当于把第层的sensitive map周围补一圈0，在与180度翻转后的filter进行cross-correlation，就能得到想要结果，如下图所示：

![캡처34](https://user-images.githubusercontent.com/60682087/89291590-c746d280-d695-11ea-9077-0f8d9e79940b.JPG)

因为卷积相当于将filter旋转180度的cross-correlation，因此上图的计算可以用卷积公式完美的表达：

![캡처35](https://user-images.githubusercontent.com/60682087/89291683-f0fff980-d695-11ea-916b-eff88c86f5cb.JPG)

![캡처36](https://user-images.githubusercontent.com/60682087/89291956-6a97e780-d696-11ea-96e1-bdaa80b87eab.JPG)

其中，符号![캡처37](https://user-images.githubusercontent.com/60682087/89292047-a03cd080-d696-11ea-89ec-c944d3a88e24.JPG)
表示element-wise product，即将矩阵中每个对应元素相乘。注意式8中的d^i-1, d^j , net^j-1都是矩阵。以上就是步长为1、输入的深度为1、filter个数为1的最简单的情况，卷积层误差项传递的算法。
下面我来推导一下步长为S的情况。卷积步长为S时的误差传递,先来看看步长为S与步长为1的差别。

![캡처38](https://user-images.githubusercontent.com/60682087/89292279-0d506600-d697-11ea-9948-a434f815777d.JPG)

如上图，上面是步长为1时的卷积结果，下面是步长为2时的卷积结果。可以看出，因为步长为2，得到的feature map跳过了步长为1时相应的部分。因此，当反向计算误差项时，可以对步长为S的sensitivity map相应的位置进行补0，将其『还原』成步长为1时的sensitivity map，再用式8进行求解。
输入层深度为D时的误差传递,当输入深度为D时，filter的深度也必须为D，j-1层的di通道只与filter的di通道的权重进行计算。因此，反向计算误差项时，可以使用式8，用filter的第di通道权重对第j层sensitivity map进行卷积，得到第j-1层di通道的sensitivity map。如下图所示：

![캡처39](https://user-images.githubusercontent.com/60682087/89292578-8780ea80-d697-11ea-8916-ba68a7e8dc20.JPG)

filter数量为N时的误差传递

filter数量为N时，输出层的深度也为N，第i个filter卷积产生输出层的第i个feature map。由于第j-1层每个加权输入![캡처40](https://user-images.githubusercontent.com/60682087/89292979-24dc1e80-d698-11ea-9ae4-d4a5a9491d81.JPG)
都同时影响了第j层所有feature map的输出值，因此，反向计算误差项时，需要使用全导数公式。也就是，先使用第d个filter对第j层相应的第d个sensitivity map进行卷积，得到一组N个j-1层的偏sensitivity map。依次用每个filter做这种卷积，就得到D组偏sensitivity map。最后在各组之间将N个偏sensitivity map 按元素相加，得到最终的N个j-1层的sensitivity map：

![캡처41](https://user-images.githubusercontent.com/60682087/89293158-640a6f80-d698-11ea-84af-00166e87dcb5.JPG)

### 卷积层filter权重梯度的计算

要在得到第j层sensitivity map的情况下，计算filter的权重的梯度，由于卷积层是权重共享的，因此梯度的计算稍有不同。

![캡처42](https://user-images.githubusercontent.com/60682087/89294161-10008a80-d69a-11ea-8e13-be747c7c418b.JPG)

如上图所示，![캡처43](https://user-images.githubusercontent.com/60682087/89294254-33c3d080-d69a-11ea-8534-ecf72274e9da.JPG)
是第j-1层的输出，wi,wj是第j层filter的权重，![캡처44](https://user-images.githubusercontent.com/60682087/89294475-87ceb500-d69a-11ea-9f4d-2ade4f6b29f7.JPG)
是第j层的sensitivity map。我们的任务是计算wi,wj的梯度，即![캡처45](https://user-images.githubusercontent.com/60682087/89294670-cd8b7d80-d69a-11ea-8cb5-965d2a4814ed.JPG)
为了计算偏导数，我们需要考察权重wi,wj对Ed的影响。权重项wi,wj通过影响![캡처46](https://user-images.githubusercontent.com/60682087/89294805-fb70c200-d69a-11ea-937c-76f6408e4244.JPG)
的值，进而影响Ed。仍然通过几个具体的例子来看权重项wi,wj对![캡처46](https://user-images.githubusercontent.com/60682087/89294805-fb70c200-d69a-11ea-937c-76f6408e4244.JPG)
的影响，然后再从中总结出规律。

![캡처47](https://user-images.githubusercontent.com/60682087/89294997-45f23e80-d69b-11ea-892e-9a4e59b4ce90.JPG)

![캡처48](https://user-images.githubusercontent.com/60682087/89295081-67532a80-d69b-11ea-8700-dadc570ceb8e.JPG)

也就是用sensitivity map作为卷积核，在input上进行cross-correlation，如下图所示：

![캡처49](https://user-images.githubusercontent.com/60682087/89295176-9073bb00-d69b-11ea-9669-e9653fe74ee2.JPG)

![캡처50](https://user-images.githubusercontent.com/60682087/89295325-c5800d80-d69b-11ea-97d5-afd3f2832bfc.JPG)

也就是偏置项的梯度就是sensitivity map所有误差项之和。对于步长为S的卷积层，处理方法与传递**误差项*是一样的，首先将sensitivity map『还原』成步长为1时的sensitivity map，再用上面的方法进行计算。
获得了所有的梯度之后，就是根据梯度下降算法来更新每个权重。这在前面的文章中已经反复写过，这里就不再重复了。至此，我已经解决了卷积层的训练问题，接下来看一看Pooling层的训练。

### Pooling层的训练

无论max pooling还是mean pooling，都没有需要学习的参数。因此，在卷积神经网络的训练中，Pooling层需要做的仅仅是将误差项传递到上一层，而没有梯度的计算。

Max Pooling误差项的传递层大小为4*4，pooling filter大小为2*2，步长为2，这样，max pooling之后，第层大小为2*2。假设第j层的d值都已经计算完毕，我现在的任务是计算第j-1层的d值。

![캡처51](https://user-images.githubusercontent.com/60682087/89297148-6243aa80-d69e-11ea-8b02-7df0a388d1c4.JPG)

![캡처52](https://user-images.githubusercontent.com/60682087/89297301-96b76680-d69e-11ea-9a74-7d1df56c9955.JPG)

![캡처53](https://user-images.githubusercontent.com/60682087/89297300-96b76680-d69e-11ea-8591-31ad5bcdb347.JPG)

发现了规律：对于max pooling，下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的误差项的值都是0。

![캡처54](https://user-images.githubusercontent.com/60682087/89297451-d0886d00-d69e-11ea-9c57-1ca7b28d1b32.JPG)

![캡처55](https://user-images.githubusercontent.com/60682087/89297456-d1b99a00-d69e-11ea-92ca-b9f3b409afce.JPG)

Mean Pooling误差项的传递

我还是用前面屡试不爽的套路，先研究一个特殊的情形，再扩展为一般规律。

![캡처56](https://user-images.githubusercontent.com/60682087/89297532-fd3c8480-d69e-11ea-99c4-9ab8bdf129a3.JPG)

![캡처57](https://user-images.githubusercontent.com/60682087/89297646-2d842300-d69f-11ea-8e73-5b9e9ecf1e3a.JPG)

![캡처58](https://user-images.githubusercontent.com/60682087/89297649-2eb55000-d69f-11ea-93fa-9e5b46029d83.JPG)

发现了规律：对于mean pooling，下一层的误差项的值会平均分配到上一层对应区块中的所有神经元。如下图所示：

![캡처59](https://user-images.githubusercontent.com/60682087/89297764-5f958500-d69f-11ea-8823-ac71ca2cd451.JPG)

上面这算法可以表达为高大上的克罗内克积(Kronecker product)的形式。

![캡처60](https://user-images.githubusercontent.com/60682087/89297857-82279e00-d69f-11ea-88d9-a7578ce69f87.JPG)

其中，是pooling层filter的大小，d^j-1 , d^j都是矩阵。

- 这个项目是我为了重新学习人工智能而做的项目。（이 프로젝트는 내가 인공지능을 다시 공부하기위해서 만든 프로젝트입니다.）
