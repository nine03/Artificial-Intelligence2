import numpy as np

def bp_sensitivity_map(self, sensitivity_array,
                       activator):
    '''
    计算传递到上一层的sensitivity map
    sensitivity_array: 本层的sensitivity map
    activator: 上一层的激活函数
    '''
    # 处理卷积步长，对原始sensitivity map进行扩展
    expanded_array = self.expand_sensitivity_map(
        sensitivity_array)
    # full卷积，对sensitivitiy map进行zero padding
    # 虽然原始输入的zero padding单元也会获得残差
    # 但这个残差不需要继续向上传递，因此就不计算了
    expanded_width = expanded_array.shape[2]
    zp = (self.input_width +
          self.filter_width - 1 - expanded_width) / 2
    padded_array = padding(expanded_array, zp)
    # 初始化delta_array，用于保存传递到上一层的
    # sensitivity map
    self.delta_array = self.create_delta_array()
    # 对于具有多个filter的卷积层来说，最终传递到上一层的
    # sensitivity map相当于所有的filter的
    # sensitivity map之和
    for f in range(self.filter_number):
        filter = self.filters[f]
        # 将filter权重翻转180度
        flipped_weights = np.array(map(
            lambda i: np.rot90(i, 2),
            filter.get_weights()))
        # 计算与一个filter对应的delta_array
        delta_array = self.create_delta_array()
        for d in range(delta_array.shape[0]):
            conv(padded_array[f], flipped_weights[d],
                 delta_array[d], 1, 0)
        self.delta_array += delta_array
    # 将计算结果与激活函数的偏导数做element-wise乘法操作
    derivative_array = np.array(self.input_array)
    element_wise_op(derivative_array,
                    activator.backward)
    self.delta_array *= derivative_array

#expand_sensitivity_map方法就是将步长为S的sensitivity map『还原』为步长为1的sensitivity map。
   def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
            self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
            self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = \
                    sensitivity_array[:,i,j]
        return expand_array
#create_delta_array是创建用来保存传递到上一层的sensitivity map的数组。
def create_delta_array(self):
    return np.zeros((self.channel_number,
                     self.input_height, self.input_width))
#计算梯度
    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

#梯度下降算法
 def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

#卷积层的梯度检查
def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    cl = ConvLayer(5,5,3,3,3,2,1,2,IdentityActivator(),0.001)
    cl.filters[0].weights = np.array(
        [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    cl.filters[0].bias=1
    cl.filters[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    return a, b, cl
def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)
    # 求取sensitivity map，是一个全1数组
    sensitivity_array = np.ones(cl.output_array.shape,
                                dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array,
                  IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d,i,j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d,i,j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d,i,j] += epsilon
                print 'weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].weights_grad[d,i,j])
