ŗÖ
­
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
ģ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ō
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.13.12
b'unknown'8¹©

conv2d_inputPlaceholder*
dtype0*$
shape:’’’’’’’’’*/
_output_shapes
:’’’’’’’’’
©
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n§®½*
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n§®=*
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
×
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
T0*
dtype0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
Ņ
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
ģ
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
Ž
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
 
conv2d/kernelVarHandleOp*
dtype0*
shared_nameconv2d/kernel*
shape: * 
_class
loc:@conv2d/kernel*
_output_shapes
: 
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 

conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0* 
_class
loc:@conv2d/kernel

!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

conv2d/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

conv2d/biasVarHandleOp*
dtype0*
shared_nameconv2d/bias*
shape: *
_class
loc:@conv2d/bias*
_output_shapes
: 
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
dtype0*
_class
loc:@conv2d/bias

conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
¦
conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*
strides
*
T0*
paddingVALID*/
_output_shapes
:’’’’’’’’’ 
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:’’’’’’’’’ 
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:’’’’’’’’’ 

max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:’’’’’’’’’ 
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"              *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ēÓz½*
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ēÓz=*
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
Ż
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
Ś
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
ō
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ę
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
¦
conv2d_1/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_1/kernel*
shape:  *"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 

conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_1/kernel

#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

conv2d_1/bias/Initializer/zerosConst*
valueB *    *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

conv2d_1/biasVarHandleOp*
dtype0*
shared_nameconv2d_1/bias*
shape: * 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 

conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_1/bias

!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:  
³
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
strides
*
T0*
paddingVALID*/
_output_shapes
:’’’’’’’’’ 
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
: 

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:’’’’’’’’’ 
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:’’’’’’’’’ 

max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:’’’’’’’’’ 
o
dropout/IdentityIdentitymax_pooling2d_1/MaxPool*
T0*/
_output_shapes
:’’’’’’’’’ 
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
:

.conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *«ŖŖ½*
dtype0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ŖŖ=*
dtype0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
Ż
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: @
Ś
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
ō
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: @
ę
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: @
¦
conv2d_2/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_2/kernel*
shape: @*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 

conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_2/kernel

#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: @

conv2d_2/bias/Initializer/zerosConst*
valueB@*    *
dtype0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@

conv2d_2/biasVarHandleOp*
dtype0*
shared_nameconv2d_2/bias*
shape:@* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 

conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_2/bias

!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
: @
®
conv2d_2/Conv2DConv2Ddropout/Identityconv2d_2/Conv2D/ReadVariableOp*
strides
*
T0*
paddingVALID*/
_output_shapes
:’’’’’’’’’@
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:’’’’’’’’’@
a
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:’’’’’’’’’@

max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:’’’’’’’’’@
­
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @      *
dtype0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
:

.conv2d_3/kernel/Initializer/random_uniform/minConst*
valueB
 *ó5¾*
dtype0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 

.conv2d_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ó5>*
dtype0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
Ž
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
T0*
dtype0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@
Ś
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
õ
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@
ē
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@
§
conv2d_3/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_3/kernel*
shape:@*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 

conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_3/kernel
 
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@

conv2d_3/bias/Initializer/zerosConst*
valueB*    *
dtype0* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:

conv2d_3/biasVarHandleOp*
dtype0*
shared_nameconv2d_3/bias*
shape:* 
_class
loc:@conv2d_3/bias*
_output_shapes
: 
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 

conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_3/bias

!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
w
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*'
_output_shapes
:@
¶
conv2d_3/Conv2DConv2Dmax_pooling2d_2/MaxPoolconv2d_3/Conv2D/ReadVariableOp*
strides
*
T0*
paddingVALID*0
_output_shapes
:’’’’’’’’’
j
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes	
:

conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:’’’’’’’’’
b
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*0
_output_shapes
:’’’’’’’’’

max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu*
strides
*
paddingVALID*
ksize
*0
_output_shapes
:’’’’’’’’’
r
dropout_1/IdentityIdentitymax_pooling2d_3/MaxPool*
T0*0
_output_shapes
:’’’’’’’’’
O
flatten/ShapeShapedropout_1/Identity*
T0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Õ
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
{
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*
N*
_output_shapes
:
x
flatten/ReshapeReshapedropout_1/Identityflatten/Reshape/shape*
T0*(
_output_shapes
:’’’’’’’’’

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *   ¾*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
Ī
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
_class
loc:@dense/kernel* 
_output_shapes
:

Ī
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
ā
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:

Ō
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:


dense/kernelVarHandleOp*
dtype0*
shared_namedense/kernel*
shape:
*
_class
loc:@dense/kernel*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0*
_class
loc:@dense/kernel

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_class
loc:@dense/kernel* 
_output_shapes
:


dense/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@dense/bias*
_output_shapes	
:


dense/biasVarHandleOp*
dtype0*
shared_name
dense/bias*
shape:*
_class
loc:@dense/bias*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
_class
loc:@dense/bias

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_class
loc:@dense/bias*
_output_shapes	
:
j
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0* 
_output_shapes
:

w
dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’
d
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:
w
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
]
dropout_2/IdentityIdentity
dense/Relu*
T0*(
_output_shapes
:’’’’’’’’’
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ŲŹ¾*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ŲŹ>*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
Ó
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ū
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


dense_1/kernelVarHandleOp*
dtype0*
shared_namedense_1/kernel*
shape:	
*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 

dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@dense_1/kernel

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


dense_1/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:


dense_1/biasVarHandleOp*
dtype0*
shared_namedense_1/bias*
shape:
*
_class
loc:@dense_1/bias*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 

dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
_class
loc:@dense_1/bias

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:

m
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	

}
dense_1/MatMulMatMuldropout_2/Identitydense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’

g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’

]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’

,
predict/group_depsNoOp^dense_1/Softmax
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_5Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_11Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_12Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_13Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_14Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_15Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
]
Const_16Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
X
Const_17Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_18Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_19Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_20Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_21Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_22Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_23Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_24Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_25Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_26Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_27Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_28Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_29Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_30Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_31Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_32Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 

RestoreV2/tensor_namesConst*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
c
RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2Const_16RestoreV2/tensor_namesRestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
J
AssignVariableOpAssignVariableOpconv2d/kernelIdentity*
dtype0

RestoreV2_1/tensor_namesConst*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_16RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
L
AssignVariableOp_1AssignVariableOpconv2d/bias
Identity_1*
dtype0

RestoreV2_2/tensor_namesConst*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_16RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpconv2d_1/kernel
Identity_2*
dtype0

RestoreV2_3/tensor_namesConst*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_16RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
N
AssignVariableOp_3AssignVariableOpconv2d_1/bias
Identity_3*
dtype0

RestoreV2_4/tensor_namesConst*K
valueBB@B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_4	RestoreV2Const_16RestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
P
AssignVariableOp_4AssignVariableOpconv2d_2/kernel
Identity_4*
dtype0

RestoreV2_5/tensor_namesConst*I
value@B>B4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2Const_16RestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
N
AssignVariableOp_5AssignVariableOpconv2d_2/bias
Identity_5*
dtype0

RestoreV2_6/tensor_namesConst*K
valueBB@B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_6	RestoreV2Const_16RestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
P
AssignVariableOp_6AssignVariableOpconv2d_3/kernel
Identity_6*
dtype0

RestoreV2_7/tensor_namesConst*I
value@B>B4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_7	RestoreV2Const_16RestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
N
AssignVariableOp_7AssignVariableOpconv2d_3/bias
Identity_7*
dtype0

RestoreV2_8/tensor_namesConst*K
valueBB@B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_8	RestoreV2Const_16RestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
M
AssignVariableOp_8AssignVariableOpdense/kernel
Identity_8*
dtype0

RestoreV2_9/tensor_namesConst*I
value@B>B4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_9	RestoreV2Const_16RestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
T0*
_output_shapes
:
K
AssignVariableOp_9AssignVariableOp
dense/bias
Identity_9*
dtype0

RestoreV2_10/tensor_namesConst*K
valueBB@B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
f
RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_10	RestoreV2Const_16RestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
Q
AssignVariableOp_10AssignVariableOpdense_1/kernelIdentity_10*
dtype0

RestoreV2_11/tensor_namesConst*I
value@B>B4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
f
RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_11	RestoreV2Const_16RestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
O
AssignVariableOp_11AssignVariableOpdense_1/biasIdentity_11*
dtype0
O
VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_1/bias*
_output_shapes
: 
S
VarIsInitializedOp_2VarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense/kernel*
_output_shapes
: 
S
VarIsInitializedOp_4VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_5VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
Q
VarIsInitializedOp_6VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_7VarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
O
VarIsInitializedOp_8VarIsInitializedOpconv2d/bias*
_output_shapes
: 
S
VarIsInitializedOp_9VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
O
VarIsInitializedOp_10VarIsInitializedOp
dense/bias*
_output_shapes
: 
S
VarIsInitializedOp_11VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
 
initNoOp^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign
X
Const_33Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_34Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
č
SaveV2/tensor_namesConst"/device:CPU:0*
valueBB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-11/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-13/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:
§
SaveV2/shape_and_slicesConst"/device:CPU:0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ņ
SaveV2SaveV2Const_34SaveV2/tensor_namesSaveV2/shape_and_slicesConst_17Const_18Const_19Const_20Const_21Const_22Const_23Const_24Const_25Const_26Const_27Const_28Const_29Const_30Const_31Const_32!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst_33"/device:CPU:0*+
dtypes!
2
Z
Identity_12IdentityConst_34^SaveV2"/device:CPU:0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
Ą
save/SaveV2/tensor_namesConst*ó

valueé
Bę
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-11/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-13/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ķ%
save/SaveV2/tensors_0Const*§%
value%B% B%{"class_name": "Sequential", "config": {"layers": [{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 28, 28, 1], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 32, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [5, 5], "name": "conv2d", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 32, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [5, 5], "name": "conv2d_1", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_1", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout", "noise_shape": null, "rate": 0.25, "seed": null, "trainable": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [3, 3], "name": "conv2d_2", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_2", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [1, 1], "name": "conv2d_3", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_3", "padding": "valid", "pool_size": [1, 1], "strides": [2, 2], "trainable": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_1", "noise_shape": null, "rate": 0.25, "seed": null, "trainable": true}}, {"class_name": "Flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}}, {"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 256, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_2", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 10, "use_bias": true}}], "name": "sequential"}}*
dtype0*
_output_shapes
: 
ē
save/SaveV2/tensors_1Const*”
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "name": "conv2d_input", "sparse": false}}*
dtype0*
_output_shapes
: 
ģ
save/SaveV2/tensors_2Const*¦
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_1", "noise_shape": null, "rate": 0.25, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
Ų
save/SaveV2/tensors_3Const*
valueB B{"class_name": "Flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}}*
dtype0*
_output_shapes
: 
ė
save/SaveV2/tensors_4Const*„
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_2", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
 
save/SaveV2/tensors_5Const*Ś
valueŠBĶ BĘ{"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}*
dtype0*
_output_shapes
: 
¢
save/SaveV2/tensors_6Const*Ü
valueŅBĻ BČ{"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_1", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}*
dtype0*
_output_shapes
: 
ź
save/SaveV2/tensors_7Const*¤
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout", "noise_shape": null, "rate": 0.25, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
¢
save/SaveV2/tensors_8Const*Ü
valueŅBĻ BČ{"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_2", "padding": "valid", "pool_size": [2, 2], "strides": [2, 2], "trainable": true}}*
dtype0*
_output_shapes
: 
¢
save/SaveV2/tensors_9Const*Ü
valueŅBĻ BČ{"class_name": "MaxPooling2D", "config": {"data_format": "channels_last", "dtype": "float32", "name": "max_pooling2d_3", "padding": "valid", "pool_size": [1, 1], "strides": [2, 2], "trainable": true}}*
dtype0*
_output_shapes
: 
Č
save/SaveV2/tensors_10Const*
value÷Bō Bķ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 28, 28, 1], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 32, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [5, 5], "name": "conv2d", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
¢
save/SaveV2/tensors_13Const*Ū
valueŃBĪ BĒ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 32, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [5, 5], "name": "conv2d_1", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
¢
save/SaveV2/tensors_16Const*Ū
valueŃBĪ BĒ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [3, 3], "name": "conv2d_2", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
£
save/SaveV2/tensors_19Const*Ü
valueŅBĻ BČ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [1, 1], "name": "conv2d_3", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
¦
save/SaveV2/tensors_22Const*ß
valueÕBŅ BĖ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 256, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ŗ
save/SaveV2/tensors_25Const*ć
valueŁBÖ BĻ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 10, "use_bias": true}}*
dtype0*
_output_shapes
: 

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/SaveV2/tensors_0save/SaveV2/tensors_1save/SaveV2/tensors_2save/SaveV2/tensors_3save/SaveV2/tensors_4save/SaveV2/tensors_5save/SaveV2/tensors_6save/SaveV2/tensors_7save/SaveV2/tensors_8save/SaveV2/tensors_9save/SaveV2/tensors_10conv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpsave/SaveV2/tensors_13!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOpsave/SaveV2/tensors_16!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOpsave/SaveV2/tensors_19!conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOpsave/SaveV2/tensors_22dense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpsave/SaveV2/tensors_25 dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp**
dtypes 
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ņ
save/RestoreV2/tensor_namesConst"/device:CPU:0*ó

valueé
Bę
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-11/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-13/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
­
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
§
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0**
dtypes 
2*
_output_shapesr
p::::::::::::::::::::::::::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp

save/NoOp_4NoOp

save/NoOp_5NoOp

save/NoOp_6NoOp

save/NoOp_7NoOp

save/NoOp_8NoOp

save/NoOp_9NoOp

save/NoOp_10NoOp
O
save/IdentityIdentitysave/RestoreV2:11*
T0*
_output_shapes
:
R
save/AssignVariableOpAssignVariableOpconv2d/biassave/Identity*
dtype0
Q
save/Identity_1Identitysave/RestoreV2:12*
T0*
_output_shapes
:
X
save/AssignVariableOp_1AssignVariableOpconv2d/kernelsave/Identity_1*
dtype0

save/NoOp_11NoOp
Q
save/Identity_2Identitysave/RestoreV2:14*
T0*
_output_shapes
:
X
save/AssignVariableOp_2AssignVariableOpconv2d_1/biassave/Identity_2*
dtype0
Q
save/Identity_3Identitysave/RestoreV2:15*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpconv2d_1/kernelsave/Identity_3*
dtype0

save/NoOp_12NoOp
Q
save/Identity_4Identitysave/RestoreV2:17*
T0*
_output_shapes
:
X
save/AssignVariableOp_4AssignVariableOpconv2d_2/biassave/Identity_4*
dtype0
Q
save/Identity_5Identitysave/RestoreV2:18*
T0*
_output_shapes
:
Z
save/AssignVariableOp_5AssignVariableOpconv2d_2/kernelsave/Identity_5*
dtype0

save/NoOp_13NoOp
Q
save/Identity_6Identitysave/RestoreV2:20*
T0*
_output_shapes
:
X
save/AssignVariableOp_6AssignVariableOpconv2d_3/biassave/Identity_6*
dtype0
Q
save/Identity_7Identitysave/RestoreV2:21*
T0*
_output_shapes
:
Z
save/AssignVariableOp_7AssignVariableOpconv2d_3/kernelsave/Identity_7*
dtype0

save/NoOp_14NoOp
Q
save/Identity_8Identitysave/RestoreV2:23*
T0*
_output_shapes
:
U
save/AssignVariableOp_8AssignVariableOp
dense/biassave/Identity_8*
dtype0
Q
save/Identity_9Identitysave/RestoreV2:24*
T0*
_output_shapes
:
W
save/AssignVariableOp_9AssignVariableOpdense/kernelsave/Identity_9*
dtype0

save/NoOp_15NoOp
R
save/Identity_10Identitysave/RestoreV2:26*
T0*
_output_shapes
:
Y
save/AssignVariableOp_10AssignVariableOpdense_1/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:27*
T0*
_output_shapes
:
[
save/AssignVariableOp_11AssignVariableOpdense_1/kernelsave/Identity_11*
dtype0
“
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
^save/NoOp^save/NoOp_1^save/NoOp_10^save/NoOp_11^save/NoOp_12^save/NoOp_13^save/NoOp_14^save/NoOp_15^save/NoOp_2^save/NoOp_3^save/NoOp_4^save/NoOp_5^save/NoOp_6^save/NoOp_7^save/NoOp_8^save/NoOp_9

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"Ń
trainable_variables¹¶
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"Ē
	variables¹¶
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08*¤
serving_default
=
conv2d_input-
conv2d_input:0’’’’’’’’’3
dense_1(
dense_1/Softmax:0’’’’’’’’’
tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1