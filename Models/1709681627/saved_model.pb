��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
x
MatrixBandPart

input"T
	num_lower"Tindex
	num_upper"Tindex	
band"T"	
Ttype"
Tindextype0	:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78́
�
1Adam/v/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/v/multi_head_attention/attention_output/bias
�
EAdam/v/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp1Adam/v/multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
�
1Adam/m/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/m/multi_head_attention/attention_output/bias
�
EAdam/m/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp1Adam/m/multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
�
3Adam/v/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/v/multi_head_attention/attention_output/kernel
�
GAdam/v/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention/attention_output/kernel*"
_output_shapes
: *
dtype0
�
3Adam/m/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/m/multi_head_attention/attention_output/kernel
�
GAdam/m/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention/attention_output/kernel*"
_output_shapes
: *
dtype0
�
&Adam/v/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/v/multi_head_attention/value/bias
�
:Adam/v/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
�
&Adam/m/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/m/multi_head_attention/value/bias
�
:Adam/m/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
�
(Adam/v/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/v/multi_head_attention/value/kernel
�
<Adam/v/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention/value/kernel*"
_output_shapes
: *
dtype0
�
(Adam/m/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/m/multi_head_attention/value/kernel
�
<Adam/m/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention/value/kernel*"
_output_shapes
: *
dtype0
�
$Adam/v/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/v/multi_head_attention/key/bias
�
8Adam/v/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp$Adam/v/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
�
$Adam/m/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/m/multi_head_attention/key/bias
�
8Adam/m/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp$Adam/m/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
�
&Adam/v/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/v/multi_head_attention/key/kernel
�
:Adam/v/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/key/kernel*"
_output_shapes
: *
dtype0
�
&Adam/m/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/m/multi_head_attention/key/kernel
�
:Adam/m/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/key/kernel*"
_output_shapes
: *
dtype0
�
&Adam/v/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/v/multi_head_attention/query/bias
�
:Adam/v/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
�
&Adam/m/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/m/multi_head_attention/query/bias
�
:Adam/m/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
�
(Adam/v/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/v/multi_head_attention/query/kernel
�
<Adam/v/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention/query/kernel*"
_output_shapes
: *
dtype0
�
(Adam/m/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/m/multi_head_attention/query/kernel
�
<Adam/m/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention/query/kernel*"
_output_shapes
: *
dtype0
�
Adam/v/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/v/layer_normalization/beta
�
3Adam/v/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/layer_normalization/beta*
_output_shapes
: *
dtype0
�
Adam/m/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/m/layer_normalization/beta
�
3Adam/m/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/layer_normalization/beta*
_output_shapes
: *
dtype0
�
 Adam/v/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/layer_normalization/gamma
�
4Adam/v/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/layer_normalization/gamma*
_output_shapes
: *
dtype0
�
 Adam/m/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/layer_normalization/gamma
�
4Adam/m/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/layer_normalization/gamma*
_output_shapes
: *
dtype0
�
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_1/beta
�
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
: *
dtype0
�
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_1/beta
�
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
: *
dtype0
�
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_1/gamma
�
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_1/gamma
�
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/v/batch_normalization/beta
�
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
: *
dtype0
�
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/m/batch_normalization/beta
�
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
: *
dtype0
�
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/batch_normalization/gamma
�
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
: *
dtype0
�
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/batch_normalization/gamma
�
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
: *
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�-*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:�-*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�-*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:�-*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �-*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	 �-*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �-*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	 �-*
dtype0
�
Adam/v/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameAdam/v/embedding_1/embeddings
�
1Adam/v/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_1/embeddings*
_output_shapes

: *
dtype0
�
Adam/m/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameAdam/m/embedding_1/embeddings
�
1Adam/m/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_1/embeddings*
_output_shapes

: *
dtype0
�
Adam/v/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�- *,
shared_nameAdam/v/embedding/embeddings
�
/Adam/v/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding/embeddings*
_output_shapes
:	�- *
dtype0
�
Adam/m/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�- *,
shared_nameAdam/m/embedding/embeddings
�
/Adam/m/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding/embeddings*
_output_shapes
:	�- *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
: *
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:*
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
: *
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:*
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
: *
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:*
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
: *
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
: *
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
: *
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�-*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�-*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �-*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	 �-*
dtype0
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

: *
dtype0
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�- *%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	�- *
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1embedding/embeddingsembedding_1/embeddings#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betalayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_55903

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
token_embedding
	pos_embedding


dense1

dense2
	norm1
	norm2

layernorm1

layernorm2
dropout1
dropout2
	multihead
lstm
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18
)19
*20
+21*
�
0
1
2
3
4
5
6
7
"8
#9
$10
%11
&12
'13
(14
)15
*16
+17*
	
,0* 
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

2trace_0
3trace_1* 

4trace_0
5trace_1* 
* 
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

embeddings*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

embeddings*

B	keras_api* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias*
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	gamma
beta
moving_mean
moving_variance*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
Vaxis
	gamma
beta
 moving_mean
!moving_variance*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	"gamma
#beta*

^	keras_api* 
(
_	keras_api
`_random_generator* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator* 
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
n_query_dense
o
_key_dense
p_value_dense
q_softmax
r_dropout_layer
s_output_dense*
B
t	keras_api
u_random_generator
vcell
w
state_spec* 
�
x
_variables
y_iterations
z_learning_rate
{_index_dict
|
_momentums
}_velocities
~_update_step_xla*

serving_default* 
TN
VARIABLE_VALUEembedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElayer_normalization/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer_normalization/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEmulti_head_attention/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*multi_head_attention/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 
 
0
1
 2
!3*
Z
0
	1

2
3
4
5
6
7
8
9
10
11*
* 
* 
* 
* 
* 
* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
	
,0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
1
 2
!3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
<
$0
%1
&2
'3
(4
)5
*6
+7*
<
$0
%1
&2
'3
(4
)5
*6
+7*
"
�0
�1
�2
�3* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

$kernel
%bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

&kernel
'bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

(kernel
)bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

*kernel
+bias*
* 
* 
;
�	keras_api
�_random_generator
�
state_size* 
* 
�
y0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
,0* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

 0
!1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
.
n0
o1
p2
q3
r4
s5*
* 
* 
* 
* 
* 
* 
* 

$0
%1*

$0
%1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

&0
'1*

&0
'1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

(0
)1*

(0
)1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

*0
+1*

*0
+1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
f`
VARIABLE_VALUEAdam/m/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/embedding_1/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/embedding_1/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/batch_normalization/gamma2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/batch_normalization/beta2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/batch_normalization/beta2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/layer_normalization/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/layer_normalization/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/layer_normalization/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/layer_normalization/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention/query/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention/query/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention/query/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention/query/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention/key/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention/key/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/multi_head_attention/key/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/multi_head_attention/key/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention/value/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention/value/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention/value/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention/value/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention/attention_output/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention/attention_output/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/m/multi_head_attention/attention_output/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/v/multi_head_attention/attention_output/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsdense_1/kerneldense_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancelayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/embedding_1/embeddingsAdam/v/embedding_1/embeddingsAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/beta"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/beta Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/beta(Adam/m/multi_head_attention/query/kernel(Adam/v/multi_head_attention/query/kernel&Adam/m/multi_head_attention/query/bias&Adam/v/multi_head_attention/query/bias&Adam/m/multi_head_attention/key/kernel&Adam/v/multi_head_attention/key/kernel$Adam/m/multi_head_attention/key/bias$Adam/v/multi_head_attention/key/bias(Adam/m/multi_head_attention/value/kernel(Adam/v/multi_head_attention/value/kernel&Adam/m/multi_head_attention/value/bias&Adam/v/multi_head_attention/value/bias3Adam/m/multi_head_attention/attention_output/kernel3Adam/v/multi_head_attention/attention_output/kernel1Adam/m/multi_head_attention/attention_output/bias1Adam/v/multi_head_attention/attention_output/biasConst*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_56849
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsdense_1/kerneldense_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancelayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/embedding_1/embeddingsAdam/v/embedding_1/embeddingsAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/beta"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/beta Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/beta(Adam/m/multi_head_attention/query/kernel(Adam/v/multi_head_attention/query/kernel&Adam/m/multi_head_attention/query/bias&Adam/v/multi_head_attention/query/bias&Adam/m/multi_head_attention/key/kernel&Adam/v/multi_head_attention/key/kernel$Adam/m/multi_head_attention/key/bias$Adam/v/multi_head_attention/key/bias(Adam/m/multi_head_attention/value/kernel(Adam/v/multi_head_attention/value/kernel&Adam/m/multi_head_attention/value/bias&Adam/v/multi_head_attention/value/bias3Adam/m/multi_head_attention/attention_output/kernel3Adam/v/multi_head_attention/attention_output/kernel1Adam/m/multi_head_attention/attention_output/bias1Adam/v/multi_head_attention/attention_output/bias*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_57038�
�'
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56145

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_56467m
Wmulti_head_attention_attention_output_kernel_regularizer_l2loss_readvariableop_resource: 
identity��Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpWmulti_head_attention_attention_output_kernel_regularizer_l2loss_readvariableop_resource*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
IdentityIdentity@multi_head_attention/attention_output/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: s
NoOpNoOpO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
'__inference_dense_1_layer_call_fn_55970

inputs
unknown:	 �-
	unknown_0:	�-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_55458t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55966:%!

_user_specified_name55964:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_56459b
Lmulti_head_attention_value_kernel_regularizer_l2loss_readvariableop_resource: 
identity��Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpLmulti_head_attention_value_kernel_regularizer_l2loss_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity5multi_head_attention/value/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: h
NoOpNoOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56085

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
3__inference_batch_normalization_layer_call_fn_56018

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55117|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name56014:%!

_user_specified_name56012:%!

_user_specified_name56010:%!

_user_specified_name56008:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_55903
input_1	
unknown:	�- 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:	 �-

unknown_20:	�-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_55083t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55899:%!

_user_specified_name55897:%!

_user_specified_name55895:%!

_user_specified_name55893:%!

_user_specified_name55891:%!

_user_specified_name55889:%!

_user_specified_name55887:%!

_user_specified_name55885:%!

_user_specified_name55883:%!

_user_specified_name55881:%!

_user_specified_name55879:%!

_user_specified_name55877:%
!

_user_specified_name55875:%	!

_user_specified_name55873:%!

_user_specified_name55871:%!

_user_specified_name55869:%!

_user_specified_name55867:%!

_user_specified_name55865:%!

_user_specified_name55863:%!

_user_specified_name55861:%!

_user_specified_name55859:%!

_user_specified_name55857:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�`
�
F__inference_jerry_model_layer_call_and_return_conditional_losses_55647
input_1	"
embedding_55488:	�- #
embedding_1_55495: '
batch_normalization_55499: '
batch_normalization_55501: '
batch_normalization_55503: '
batch_normalization_55505: '
layer_normalization_55508: '
layer_normalization_55510: 0
multi_head_attention_55589: ,
multi_head_attention_55591:0
multi_head_attention_55593: ,
multi_head_attention_55595:0
multi_head_attention_55597: ,
multi_head_attention_55599:0
multi_head_attention_55601: (
multi_head_attention_55603: )
batch_normalization_1_55606: )
batch_normalization_1_55608: )
batch_normalization_1_55610: )
batch_normalization_1_55612:  
dense_1_55621:	 �-
dense_1_55623:	�-
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_55488*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55253M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallrange:output:0embedding_1_55495*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_55268�
addAddV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:��������� �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCalladd:z:0batch_normalization_55499batch_normalization_55501batch_normalization_55503batch_normalization_55505*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55137�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0layer_normalization_55508layer_normalization_55510*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_55303�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_55589multi_head_attention_55591multi_head_attention_55593multi_head_attention_55595multi_head_attention_55597multi_head_attention_55599multi_head_attention_55601multi_head_attention_55603*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55588�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0batch_normalization_1_55606batch_normalization_1_55608batch_normalization_1_55610batch_normalization_1_55612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55217�
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_55619�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_55621dense_1_55623*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_55458
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_55621*
_output_shapes
:	 �-*
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55589*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55593*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55597*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55601*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCallO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:%!

_user_specified_name55623:%!

_user_specified_name55621:%!

_user_specified_name55612:%!

_user_specified_name55610:%!

_user_specified_name55608:%!

_user_specified_name55606:%!

_user_specified_name55603:%!

_user_specified_name55601:%!

_user_specified_name55599:%!

_user_specified_name55597:%!

_user_specified_name55595:%!

_user_specified_name55593:%
!

_user_specified_name55591:%	!

_user_specified_name55589:%!

_user_specified_name55510:%!

_user_specified_name55508:%!

_user_specified_name55505:%!

_user_specified_name55503:%!

_user_specified_name55501:%!

_user_specified_name55499:%!

_user_specified_name55495:%!

_user_specified_name55488:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�'
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55197

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_55253

inputs	)
embedding_lookup_55248:	�- 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55248inputs*
Tindices0	*)
_class
loc:@embedding_lookup/55248*+
_output_shapes
:��������� *
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:��������� u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:%!

_user_specified_name55248:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_56223

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:��������� _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55217

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
~
)__inference_embedding_layer_call_fn_55938

inputs	
unknown:	�- 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55253s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55934:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_layer_normalization_layer_call_fn_56174

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_55303s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name56170:%!

_user_specified_name56168:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_55931L
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:	 �-
identity��0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	 �-*
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: U
NoOpNoOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
��
�+
!__inference__traced_restore_57038
file_prefix8
%assignvariableop_embedding_embeddings:	�- ;
)assignvariableop_1_embedding_1_embeddings: 4
!assignvariableop_2_dense_1_kernel:	 �-.
assignvariableop_3_dense_1_bias:	�-:
,assignvariableop_4_batch_normalization_gamma: 9
+assignvariableop_5_batch_normalization_beta: @
2assignvariableop_6_batch_normalization_moving_mean: D
6assignvariableop_7_batch_normalization_moving_variance: <
.assignvariableop_8_batch_normalization_1_gamma: ;
-assignvariableop_9_batch_normalization_1_beta: C
5assignvariableop_10_batch_normalization_1_moving_mean: G
9assignvariableop_11_batch_normalization_1_moving_variance: ;
-assignvariableop_12_layer_normalization_gamma: :
,assignvariableop_13_layer_normalization_beta: K
5assignvariableop_14_multi_head_attention_query_kernel: E
3assignvariableop_15_multi_head_attention_query_bias:I
3assignvariableop_16_multi_head_attention_key_kernel: C
1assignvariableop_17_multi_head_attention_key_bias:K
5assignvariableop_18_multi_head_attention_value_kernel: E
3assignvariableop_19_multi_head_attention_value_bias:V
@assignvariableop_20_multi_head_attention_attention_output_kernel: L
>assignvariableop_21_multi_head_attention_attention_output_bias: '
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: B
/assignvariableop_24_adam_m_embedding_embeddings:	�- B
/assignvariableop_25_adam_v_embedding_embeddings:	�- C
1assignvariableop_26_adam_m_embedding_1_embeddings: C
1assignvariableop_27_adam_v_embedding_1_embeddings: <
)assignvariableop_28_adam_m_dense_1_kernel:	 �-<
)assignvariableop_29_adam_v_dense_1_kernel:	 �-6
'assignvariableop_30_adam_m_dense_1_bias:	�-6
'assignvariableop_31_adam_v_dense_1_bias:	�-B
4assignvariableop_32_adam_m_batch_normalization_gamma: B
4assignvariableop_33_adam_v_batch_normalization_gamma: A
3assignvariableop_34_adam_m_batch_normalization_beta: A
3assignvariableop_35_adam_v_batch_normalization_beta: D
6assignvariableop_36_adam_m_batch_normalization_1_gamma: D
6assignvariableop_37_adam_v_batch_normalization_1_gamma: C
5assignvariableop_38_adam_m_batch_normalization_1_beta: C
5assignvariableop_39_adam_v_batch_normalization_1_beta: B
4assignvariableop_40_adam_m_layer_normalization_gamma: B
4assignvariableop_41_adam_v_layer_normalization_gamma: A
3assignvariableop_42_adam_m_layer_normalization_beta: A
3assignvariableop_43_adam_v_layer_normalization_beta: R
<assignvariableop_44_adam_m_multi_head_attention_query_kernel: R
<assignvariableop_45_adam_v_multi_head_attention_query_kernel: L
:assignvariableop_46_adam_m_multi_head_attention_query_bias:L
:assignvariableop_47_adam_v_multi_head_attention_query_bias:P
:assignvariableop_48_adam_m_multi_head_attention_key_kernel: P
:assignvariableop_49_adam_v_multi_head_attention_key_kernel: J
8assignvariableop_50_adam_m_multi_head_attention_key_bias:J
8assignvariableop_51_adam_v_multi_head_attention_key_bias:R
<assignvariableop_52_adam_m_multi_head_attention_value_kernel: R
<assignvariableop_53_adam_v_multi_head_attention_value_kernel: L
:assignvariableop_54_adam_m_multi_head_attention_value_bias:L
:assignvariableop_55_adam_v_multi_head_attention_value_bias:]
Gassignvariableop_56_adam_m_multi_head_attention_attention_output_kernel: ]
Gassignvariableop_57_adam_v_multi_head_attention_attention_output_kernel: S
Eassignvariableop_58_adam_m_multi_head_attention_attention_output_bias: S
Eassignvariableop_59_adam_v_multi_head_attention_attention_output_bias: 
identity_61��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_layer_normalization_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_layer_normalization_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp5assignvariableop_14_multi_head_attention_query_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_multi_head_attention_query_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_multi_head_attention_key_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_multi_head_attention_key_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_multi_head_attention_value_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp3assignvariableop_19_multi_head_attention_value_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp@assignvariableop_20_multi_head_attention_attention_output_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp>assignvariableop_21_multi_head_attention_attention_output_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_m_embedding_embeddingsIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp/assignvariableop_25_adam_v_embedding_embeddingsIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_adam_m_embedding_1_embeddingsIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_v_embedding_1_embeddingsIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_dense_1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_dense_1_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_m_dense_1_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_v_dense_1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_m_batch_normalization_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_v_batch_normalization_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_m_batch_normalization_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_v_batch_normalization_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_m_batch_normalization_1_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_v_batch_normalization_1_gammaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_m_batch_normalization_1_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_v_batch_normalization_1_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_m_layer_normalization_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_v_layer_normalization_gammaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_m_layer_normalization_betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp3assignvariableop_43_adam_v_layer_normalization_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp<assignvariableop_44_adam_m_multi_head_attention_query_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_v_multi_head_attention_query_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp:assignvariableop_46_adam_m_multi_head_attention_query_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp:assignvariableop_47_adam_v_multi_head_attention_query_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp:assignvariableop_48_adam_m_multi_head_attention_key_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp:assignvariableop_49_adam_v_multi_head_attention_key_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_m_multi_head_attention_key_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_v_multi_head_attention_key_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp<assignvariableop_52_adam_m_multi_head_attention_value_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp<assignvariableop_53_adam_v_multi_head_attention_value_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp:assignvariableop_54_adam_m_multi_head_attention_value_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp:assignvariableop_55_adam_v_multi_head_attention_value_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpGassignvariableop_56_adam_m_multi_head_attention_attention_output_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpGassignvariableop_57_adam_v_multi_head_attention_attention_output_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpEassignvariableop_58_adam_m_multi_head_attention_attention_output_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpEassignvariableop_59_adam_v_multi_head_attention_attention_output_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:Q<M
K
_user_specified_name31Adam/v/multi_head_attention/attention_output/bias:Q;M
K
_user_specified_name31Adam/m/multi_head_attention/attention_output/bias:S:O
M
_user_specified_name53Adam/v/multi_head_attention/attention_output/kernel:S9O
M
_user_specified_name53Adam/m/multi_head_attention/attention_output/kernel:F8B
@
_user_specified_name(&Adam/v/multi_head_attention/value/bias:F7B
@
_user_specified_name(&Adam/m/multi_head_attention/value/bias:H6D
B
_user_specified_name*(Adam/v/multi_head_attention/value/kernel:H5D
B
_user_specified_name*(Adam/m/multi_head_attention/value/kernel:D4@
>
_user_specified_name&$Adam/v/multi_head_attention/key/bias:D3@
>
_user_specified_name&$Adam/m/multi_head_attention/key/bias:F2B
@
_user_specified_name(&Adam/v/multi_head_attention/key/kernel:F1B
@
_user_specified_name(&Adam/m/multi_head_attention/key/kernel:F0B
@
_user_specified_name(&Adam/v/multi_head_attention/query/bias:F/B
@
_user_specified_name(&Adam/m/multi_head_attention/query/bias:H.D
B
_user_specified_name*(Adam/v/multi_head_attention/query/kernel:H-D
B
_user_specified_name*(Adam/m/multi_head_attention/query/kernel:?,;
9
_user_specified_name!Adam/v/layer_normalization/beta:?+;
9
_user_specified_name!Adam/m/layer_normalization/beta:@*<
:
_user_specified_name" Adam/v/layer_normalization/gamma:@)<
:
_user_specified_name" Adam/m/layer_normalization/gamma:A(=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A'=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B&>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B%>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:?$;
9
_user_specified_name!Adam/v/batch_normalization/beta:?#;
9
_user_specified_name!Adam/m/batch_normalization/beta:@"<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@!<
:
_user_specified_name" Adam/m/batch_normalization/gamma:3 /
-
_user_specified_nameAdam/v/dense_1/bias:3/
-
_user_specified_nameAdam/m/dense_1/bias:51
/
_user_specified_nameAdam/v/dense_1/kernel:51
/
_user_specified_nameAdam/m/dense_1/kernel:=9
7
_user_specified_nameAdam/v/embedding_1/embeddings:=9
7
_user_specified_nameAdam/m/embedding_1/embeddings:;7
5
_user_specified_nameAdam/v/embedding/embeddings:;7
5
_user_specified_nameAdam/m/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:84
2
_user_specified_namelayer_normalization/beta:95
3
_user_specified_namelayer_normalization/gamma:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:62
0
_user_specified_nameembedding_1/embeddings:40
.
_user_specified_nameembedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
E
)__inference_dropout_1_layer_call_fn_56206

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_55619d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

+__inference_embedding_1_layer_call_fn_55953

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_55268f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55949:B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_56451`
Jmulti_head_attention_key_kernel_regularizer_l2loss_readvariableop_resource: 
identity��Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpJmulti_head_attention_key_kernel_regularizer_l2loss_readvariableop_resource*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: q
IdentityIdentity3multi_head_attention/key/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: f
NoOpNoOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_55946

inputs	)
embedding_lookup_55941:	�- 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55941inputs*
Tindices0	*)
_class
loc:@embedding_lookup/55941*+
_output_shapes
:��������� *
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:��������� u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:%!

_user_specified_name55941:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_55083
input_1	?
,jerry_model_embedding_embedding_lookup_54936:	�- @
.jerry_model_embedding_1_embedding_lookup_54944: O
Ajerry_model_batch_normalization_batchnorm_readvariableop_resource: S
Ejerry_model_batch_normalization_batchnorm_mul_readvariableop_resource: Q
Cjerry_model_batch_normalization_batchnorm_readvariableop_1_resource: Q
Cjerry_model_batch_normalization_batchnorm_readvariableop_2_resource: S
Ejerry_model_layer_normalization_batchnorm_mul_readvariableop_resource: O
Ajerry_model_layer_normalization_batchnorm_readvariableop_resource: b
Ljerry_model_multi_head_attention_query_einsum_einsum_readvariableop_resource: T
Bjerry_model_multi_head_attention_query_add_readvariableop_resource:`
Jjerry_model_multi_head_attention_key_einsum_einsum_readvariableop_resource: R
@jerry_model_multi_head_attention_key_add_readvariableop_resource:b
Ljerry_model_multi_head_attention_value_einsum_einsum_readvariableop_resource: T
Bjerry_model_multi_head_attention_value_add_readvariableop_resource:m
Wjerry_model_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: [
Mjerry_model_multi_head_attention_attention_output_add_readvariableop_resource: Q
Cjerry_model_batch_normalization_1_batchnorm_readvariableop_resource: U
Gjerry_model_batch_normalization_1_batchnorm_mul_readvariableop_resource: S
Ejerry_model_batch_normalization_1_batchnorm_readvariableop_1_resource: S
Ejerry_model_batch_normalization_1_batchnorm_readvariableop_2_resource: H
5jerry_model_dense_1_tensordot_readvariableop_resource:	 �-B
3jerry_model_dense_1_biasadd_readvariableop_resource:	�-
identity��8jerry_model/batch_normalization/batchnorm/ReadVariableOp�:jerry_model/batch_normalization/batchnorm/ReadVariableOp_1�:jerry_model/batch_normalization/batchnorm/ReadVariableOp_2�<jerry_model/batch_normalization/batchnorm/mul/ReadVariableOp�:jerry_model/batch_normalization_1/batchnorm/ReadVariableOp�<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_1�<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_2�>jerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOp�*jerry_model/dense_1/BiasAdd/ReadVariableOp�,jerry_model/dense_1/Tensordot/ReadVariableOp�&jerry_model/embedding/embedding_lookup�(jerry_model/embedding_1/embedding_lookup�8jerry_model/layer_normalization/batchnorm/ReadVariableOp�<jerry_model/layer_normalization/batchnorm/mul/ReadVariableOp�Djerry_model/multi_head_attention/attention_output/add/ReadVariableOp�Njerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�7jerry_model/multi_head_attention/key/add/ReadVariableOp�Ajerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOp�9jerry_model/multi_head_attention/query/add/ReadVariableOp�Cjerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOp�9jerry_model/multi_head_attention/value/add/ReadVariableOp�Cjerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOp�
&jerry_model/embedding/embedding_lookupResourceGather,jerry_model_embedding_embedding_lookup_54936input_1*
Tindices0	*?
_class5
31loc:@jerry_model/embedding/embedding_lookup/54936*+
_output_shapes
:��������� *
dtype0�
/jerry_model/embedding/embedding_lookup/IdentityIdentity/jerry_model/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:��������� Y
jerry_model/range/startConst*
_output_shapes
: *
dtype0*
value	B : Y
jerry_model/range/limitConst*
_output_shapes
: *
dtype0*
value	B :Y
jerry_model/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
jerry_model/rangeRange jerry_model/range/start:output:0 jerry_model/range/limit:output:0 jerry_model/range/delta:output:0*
_output_shapes
:�
(jerry_model/embedding_1/embedding_lookupResourceGather.jerry_model_embedding_1_embedding_lookup_54944jerry_model/range:output:0*
Tindices0*A
_class7
53loc:@jerry_model/embedding_1/embedding_lookup/54944*
_output_shapes

: *
dtype0�
1jerry_model/embedding_1/embedding_lookup/IdentityIdentity1jerry_model/embedding_1/embedding_lookup:output:0*
T0*
_output_shapes

: �
jerry_model/addAddV28jerry_model/embedding/embedding_lookup/Identity:output:0:jerry_model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:��������� �
8jerry_model/batch_normalization/batchnorm/ReadVariableOpReadVariableOpAjerry_model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0t
/jerry_model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-jerry_model/batch_normalization/batchnorm/addAddV2@jerry_model/batch_normalization/batchnorm/ReadVariableOp:value:08jerry_model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
/jerry_model/batch_normalization/batchnorm/RsqrtRsqrt1jerry_model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: �
<jerry_model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpEjerry_model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
-jerry_model/batch_normalization/batchnorm/mulMul3jerry_model/batch_normalization/batchnorm/Rsqrt:y:0Djerry_model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
/jerry_model/batch_normalization/batchnorm/mul_1Muljerry_model/add:z:01jerry_model/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� �
:jerry_model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpCjerry_model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
/jerry_model/batch_normalization/batchnorm/mul_2MulBjerry_model/batch_normalization/batchnorm/ReadVariableOp_1:value:01jerry_model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: �
:jerry_model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpCjerry_model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
-jerry_model/batch_normalization/batchnorm/subSubBjerry_model/batch_normalization/batchnorm/ReadVariableOp_2:value:03jerry_model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
/jerry_model/batch_normalization/batchnorm/add_1AddV23jerry_model/batch_normalization/batchnorm/mul_1:z:01jerry_model/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:��������� �
>jerry_model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,jerry_model/layer_normalization/moments/meanMean3jerry_model/batch_normalization/batchnorm/add_1:z:0Gjerry_model/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
4jerry_model/layer_normalization/moments/StopGradientStopGradient5jerry_model/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:����������
9jerry_model/layer_normalization/moments/SquaredDifferenceSquaredDifference3jerry_model/batch_normalization/batchnorm/add_1:z:0=jerry_model/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:��������� �
Bjerry_model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
0jerry_model/layer_normalization/moments/varianceMean=jerry_model/layer_normalization/moments/SquaredDifference:z:0Kjerry_model/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(t
/jerry_model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
-jerry_model/layer_normalization/batchnorm/addAddV29jerry_model/layer_normalization/moments/variance:output:08jerry_model/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
/jerry_model/layer_normalization/batchnorm/RsqrtRsqrt1jerry_model/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
<jerry_model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpEjerry_model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
-jerry_model/layer_normalization/batchnorm/mulMul3jerry_model/layer_normalization/batchnorm/Rsqrt:y:0Djerry_model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
/jerry_model/layer_normalization/batchnorm/mul_1Mul3jerry_model/batch_normalization/batchnorm/add_1:z:01jerry_model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� �
/jerry_model/layer_normalization/batchnorm/mul_2Mul5jerry_model/layer_normalization/moments/mean:output:01jerry_model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� �
8jerry_model/layer_normalization/batchnorm/ReadVariableOpReadVariableOpAjerry_model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
-jerry_model/layer_normalization/batchnorm/subSub@jerry_model/layer_normalization/batchnorm/ReadVariableOp:value:03jerry_model/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:��������� �
/jerry_model/layer_normalization/batchnorm/add_1AddV23jerry_model/layer_normalization/batchnorm/mul_1:z:01jerry_model/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:��������� �
&jerry_model/multi_head_attention/ShapeShape3jerry_model/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��~
4jerry_model/multi_head_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
6jerry_model/multi_head_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6jerry_model/multi_head_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.jerry_model/multi_head_attention/strided_sliceStridedSlice/jerry_model/multi_head_attention/Shape:output:0=jerry_model/multi_head_attention/strided_slice/stack:output:0?jerry_model/multi_head_attention/strided_slice/stack_1:output:0?jerry_model/multi_head_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(jerry_model/multi_head_attention/Shape_1Shape3jerry_model/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
6jerry_model/multi_head_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8jerry_model/multi_head_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8jerry_model/multi_head_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0jerry_model/multi_head_attention/strided_slice_1StridedSlice1jerry_model/multi_head_attention/Shape_1:output:0?jerry_model/multi_head_attention/strided_slice_1/stack:output:0Ajerry_model/multi_head_attention/strided_slice_1/stack_1:output:0Ajerry_model/multi_head_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.jerry_model/multi_head_attention/ones/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
,jerry_model/multi_head_attention/ones/packedPack7jerry_model/multi_head_attention/ones/packed/0:output:07jerry_model/multi_head_attention/strided_slice:output:09jerry_model/multi_head_attention/strided_slice_1:output:0*
N*
T0*
_output_shapes
:m
+jerry_model/multi_head_attention/ones/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
%jerry_model/multi_head_attention/onesFill5jerry_model/multi_head_attention/ones/packed:output:04jerry_model/multi_head_attention/ones/Const:output:0*
T0
*"
_output_shapes
:�
9jerry_model/multi_head_attention/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������{
9jerry_model/multi_head_attention/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
/jerry_model/multi_head_attention/MatrixBandPartMatrixBandPart.jerry_model/multi_head_attention/ones:output:0Bjerry_model/multi_head_attention/MatrixBandPart/num_lower:output:0Bjerry_model/multi_head_attention/MatrixBandPart/num_upper:output:0*
Tindex0*
T0
*"
_output_shapes
:�
Cjerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpLjerry_model_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4jerry_model/multi_head_attention/query/einsum/EinsumEinsum3jerry_model/layer_normalization/batchnorm/add_1:z:0Kjerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
9jerry_model/multi_head_attention/query/add/ReadVariableOpReadVariableOpBjerry_model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
*jerry_model/multi_head_attention/query/addAddV2=jerry_model/multi_head_attention/query/einsum/Einsum:output:0Ajerry_model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ajerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpJjerry_model_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
2jerry_model/multi_head_attention/key/einsum/EinsumEinsum3jerry_model/layer_normalization/batchnorm/add_1:z:0Ijerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
7jerry_model/multi_head_attention/key/add/ReadVariableOpReadVariableOp@jerry_model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
(jerry_model/multi_head_attention/key/addAddV2;jerry_model/multi_head_attention/key/einsum/Einsum:output:0?jerry_model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Cjerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpLjerry_model_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4jerry_model/multi_head_attention/value/einsum/EinsumEinsum3jerry_model/layer_normalization/batchnorm/add_1:z:0Kjerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
9jerry_model/multi_head_attention/value/add/ReadVariableOpReadVariableOpBjerry_model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
*jerry_model/multi_head_attention/value/addAddV2=jerry_model/multi_head_attention/value/einsum/Einsum:output:0Ajerry_model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������k
&jerry_model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
$jerry_model/multi_head_attention/MulMul.jerry_model/multi_head_attention/query/add:z:0/jerry_model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:����������
.jerry_model/multi_head_attention/einsum/EinsumEinsum,jerry_model/multi_head_attention/key/add:z:0(jerry_model/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbez
/jerry_model/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
+jerry_model/multi_head_attention/ExpandDims
ExpandDims6jerry_model/multi_head_attention/MatrixBandPart:band:08jerry_model/multi_head_attention/ExpandDims/dim:output:0*
T0
*&
_output_shapes
:�
-jerry_model/multi_head_attention/softmax/CastCast4jerry_model/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0
*&
_output_shapes
:s
.jerry_model/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,jerry_model/multi_head_attention/softmax/subSub7jerry_model/multi_head_attention/softmax/sub/x:output:01jerry_model/multi_head_attention/softmax/Cast:y:0*
T0*&
_output_shapes
:s
.jerry_model/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
,jerry_model/multi_head_attention/softmax/mulMul0jerry_model/multi_head_attention/softmax/sub:z:07jerry_model/multi_head_attention/softmax/mul/y:output:0*
T0*&
_output_shapes
:�
,jerry_model/multi_head_attention/softmax/addAddV27jerry_model/multi_head_attention/einsum/Einsum:output:00jerry_model/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:����������
0jerry_model/multi_head_attention/softmax/SoftmaxSoftmax0jerry_model/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:����������
1jerry_model/multi_head_attention/dropout/IdentityIdentity:jerry_model/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
0jerry_model/multi_head_attention/einsum_1/EinsumEinsum:jerry_model/multi_head_attention/dropout/Identity:output:0.jerry_model/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Njerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpWjerry_model_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
?jerry_model/multi_head_attention/attention_output/einsum/EinsumEinsum9jerry_model/multi_head_attention/einsum_1/Einsum:output:0Vjerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:��������� *
equationabcd,cde->abe�
Djerry_model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpMjerry_model_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0�
5jerry_model/multi_head_attention/attention_output/addAddV2Hjerry_model/multi_head_attention/attention_output/einsum/Einsum:output:0Ljerry_model/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
:jerry_model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpCjerry_model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0v
1jerry_model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/jerry_model/batch_normalization_1/batchnorm/addAddV2Bjerry_model/batch_normalization_1/batchnorm/ReadVariableOp:value:0:jerry_model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
1jerry_model/batch_normalization_1/batchnorm/RsqrtRsqrt3jerry_model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: �
>jerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpGjerry_model_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
/jerry_model/batch_normalization_1/batchnorm/mulMul5jerry_model/batch_normalization_1/batchnorm/Rsqrt:y:0Fjerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
1jerry_model/batch_normalization_1/batchnorm/mul_1Mul9jerry_model/multi_head_attention/attention_output/add:z:03jerry_model/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� �
<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpEjerry_model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
1jerry_model/batch_normalization_1/batchnorm/mul_2MulDjerry_model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:03jerry_model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: �
<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpEjerry_model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
/jerry_model/batch_normalization_1/batchnorm/subSubDjerry_model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:05jerry_model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
1jerry_model/batch_normalization_1/batchnorm/add_1AddV25jerry_model/batch_normalization_1/batchnorm/mul_1:z:03jerry_model/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:��������� �
jerry_model/dropout_1/IdentityIdentity5jerry_model/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:��������� �
,jerry_model/dense_1/Tensordot/ReadVariableOpReadVariableOp5jerry_model_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	 �-*
dtype0l
"jerry_model/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"jerry_model/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
#jerry_model/dense_1/Tensordot/ShapeShape'jerry_model/dropout_1/Identity:output:0*
T0*
_output_shapes
::��m
+jerry_model/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&jerry_model/dense_1/Tensordot/GatherV2GatherV2,jerry_model/dense_1/Tensordot/Shape:output:0+jerry_model/dense_1/Tensordot/free:output:04jerry_model/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-jerry_model/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(jerry_model/dense_1/Tensordot/GatherV2_1GatherV2,jerry_model/dense_1/Tensordot/Shape:output:0+jerry_model/dense_1/Tensordot/axes:output:06jerry_model/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#jerry_model/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
"jerry_model/dense_1/Tensordot/ProdProd/jerry_model/dense_1/Tensordot/GatherV2:output:0,jerry_model/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%jerry_model/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
$jerry_model/dense_1/Tensordot/Prod_1Prod1jerry_model/dense_1/Tensordot/GatherV2_1:output:0.jerry_model/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)jerry_model/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$jerry_model/dense_1/Tensordot/concatConcatV2+jerry_model/dense_1/Tensordot/free:output:0+jerry_model/dense_1/Tensordot/axes:output:02jerry_model/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
#jerry_model/dense_1/Tensordot/stackPack+jerry_model/dense_1/Tensordot/Prod:output:0-jerry_model/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
'jerry_model/dense_1/Tensordot/transpose	Transpose'jerry_model/dropout_1/Identity:output:0-jerry_model/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� �
%jerry_model/dense_1/Tensordot/ReshapeReshape+jerry_model/dense_1/Tensordot/transpose:y:0,jerry_model/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
$jerry_model/dense_1/Tensordot/MatMulMatMul.jerry_model/dense_1/Tensordot/Reshape:output:04jerry_model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������-p
%jerry_model/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�-m
+jerry_model/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&jerry_model/dense_1/Tensordot/concat_1ConcatV2/jerry_model/dense_1/Tensordot/GatherV2:output:0.jerry_model/dense_1/Tensordot/Const_2:output:04jerry_model/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
jerry_model/dense_1/TensordotReshape.jerry_model/dense_1/Tensordot/MatMul:product:0/jerry_model/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������-�
*jerry_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp3jerry_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�-*
dtype0�
jerry_model/dense_1/BiasAddBiasAdd&jerry_model/dense_1/Tensordot:output:02jerry_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������-�
jerry_model/dense_1/SoftmaxSoftmax$jerry_model/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������-y
IdentityIdentity%jerry_model/dense_1/Softmax:softmax:0^NoOp*
T0*,
_output_shapes
:����������-�

NoOpNoOp9^jerry_model/batch_normalization/batchnorm/ReadVariableOp;^jerry_model/batch_normalization/batchnorm/ReadVariableOp_1;^jerry_model/batch_normalization/batchnorm/ReadVariableOp_2=^jerry_model/batch_normalization/batchnorm/mul/ReadVariableOp;^jerry_model/batch_normalization_1/batchnorm/ReadVariableOp=^jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_1=^jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_2?^jerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOp+^jerry_model/dense_1/BiasAdd/ReadVariableOp-^jerry_model/dense_1/Tensordot/ReadVariableOp'^jerry_model/embedding/embedding_lookup)^jerry_model/embedding_1/embedding_lookup9^jerry_model/layer_normalization/batchnorm/ReadVariableOp=^jerry_model/layer_normalization/batchnorm/mul/ReadVariableOpE^jerry_model/multi_head_attention/attention_output/add/ReadVariableOpO^jerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp8^jerry_model/multi_head_attention/key/add/ReadVariableOpB^jerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOp:^jerry_model/multi_head_attention/query/add/ReadVariableOpD^jerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOp:^jerry_model/multi_head_attention/value/add/ReadVariableOpD^jerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 2x
:jerry_model/batch_normalization/batchnorm/ReadVariableOp_1:jerry_model/batch_normalization/batchnorm/ReadVariableOp_12x
:jerry_model/batch_normalization/batchnorm/ReadVariableOp_2:jerry_model/batch_normalization/batchnorm/ReadVariableOp_22t
8jerry_model/batch_normalization/batchnorm/ReadVariableOp8jerry_model/batch_normalization/batchnorm/ReadVariableOp2|
<jerry_model/batch_normalization/batchnorm/mul/ReadVariableOp<jerry_model/batch_normalization/batchnorm/mul/ReadVariableOp2|
<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_1<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_12|
<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_2<jerry_model/batch_normalization_1/batchnorm/ReadVariableOp_22x
:jerry_model/batch_normalization_1/batchnorm/ReadVariableOp:jerry_model/batch_normalization_1/batchnorm/ReadVariableOp2�
>jerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOp>jerry_model/batch_normalization_1/batchnorm/mul/ReadVariableOp2X
*jerry_model/dense_1/BiasAdd/ReadVariableOp*jerry_model/dense_1/BiasAdd/ReadVariableOp2\
,jerry_model/dense_1/Tensordot/ReadVariableOp,jerry_model/dense_1/Tensordot/ReadVariableOp2P
&jerry_model/embedding/embedding_lookup&jerry_model/embedding/embedding_lookup2T
(jerry_model/embedding_1/embedding_lookup(jerry_model/embedding_1/embedding_lookup2t
8jerry_model/layer_normalization/batchnorm/ReadVariableOp8jerry_model/layer_normalization/batchnorm/ReadVariableOp2|
<jerry_model/layer_normalization/batchnorm/mul/ReadVariableOp<jerry_model/layer_normalization/batchnorm/mul/ReadVariableOp2�
Djerry_model/multi_head_attention/attention_output/add/ReadVariableOpDjerry_model/multi_head_attention/attention_output/add/ReadVariableOp2�
Njerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpNjerry_model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2r
7jerry_model/multi_head_attention/key/add/ReadVariableOp7jerry_model/multi_head_attention/key/add/ReadVariableOp2�
Ajerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOpAjerry_model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2v
9jerry_model/multi_head_attention/query/add/ReadVariableOp9jerry_model/multi_head_attention/query/add/ReadVariableOp2�
Cjerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOpCjerry_model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2v
9jerry_model/multi_head_attention/value/add/ReadVariableOp9jerry_model/multi_head_attention/value/add/ReadVariableOp2�
Cjerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOpCjerry_model/multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:%!

_user_specified_name54944:%!

_user_specified_name54936:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�a
�	
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56343	
query	
valueA
+query_einsum_einsum_readvariableop_resource: 3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource: 1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource: 3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource: :
,attention_output_add_readvariableop_resource: 
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOpH
ShapeShapequery*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
Shape_1Shapevalue*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
ones/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
ones/packedPackones/packed/0:output:0strided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zd
onesFillones/packed:output:0ones/Const:output:0*
T0
*"
_output_shapes
:c
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������Z
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
MatrixBandPartMatrixBandPartones:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
Tindex0*
T0
*"
_output_shapes
:�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:����������
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbeY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������y

ExpandDims
ExpandDimsMatrixBandPart:band:0ExpandDims/dim:output:0*
T0
*&
_output_shapes
:i
softmax/CastCastExpandDims:output:0*

DstT0*

SrcT0
*&
_output_shapes
:R
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*&
_output_shapes
:R
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn�l
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*&
_output_shapes
:w
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*/
_output_shapes
:���������e
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:��������� *
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOpO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�#
�
B__inference_dense_1_layer_call_and_return_conditional_losses_55458

inputs4
!tensordot_readvariableop_resource:	 �-.
biasadd_readvariableop_resource:	�-
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �-*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� �
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������-\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�-Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������-s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�-*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������-[
SoftmaxSoftmaxBiasAdd:output:0*
T0*,
_output_shapes
:����������-�
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �-*
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*,
_output_shapes
:����������-�
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�&
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55117

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_55303

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:��������� f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:��������� \
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_jerry_model_layer_call_fn_55745
input_1	
unknown:	�- 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:	 �-

unknown_20:	�-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_jerry_model_layer_call_and_return_conditional_losses_55647t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55741:%!

_user_specified_name55739:%!

_user_specified_name55737:%!

_user_specified_name55735:%!

_user_specified_name55733:%!

_user_specified_name55731:%!

_user_specified_name55729:%!

_user_specified_name55727:%!

_user_specified_name55725:%!

_user_specified_name55723:%!

_user_specified_name55721:%!

_user_specified_name55719:%
!

_user_specified_name55717:%	!

_user_specified_name55715:%!

_user_specified_name55713:%!

_user_specified_name55711:%!

_user_specified_name55709:%!

_user_specified_name55707:%!

_user_specified_name55705:%!

_user_specified_name55703:%!

_user_specified_name55701:%!

_user_specified_name55699:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_55268

inputs(
embedding_lookup_55263: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55263inputs*
Tindices0*)
_class
loc:@embedding_lookup/55263*
_output_shapes

: *
dtype0i
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*
_output_shapes

: h
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*
_output_shapes

: 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:: 2$
embedding_lookupembedding_lookup:%!

_user_specified_name55263:B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_56443b
Lmulti_head_attention_query_kernel_regularizer_l2loss_readvariableop_resource: 
identity��Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpLmulti_head_attention_query_kernel_regularizer_l2loss_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity5multi_head_attention/query/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: h
NoOpNoOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�#
�
B__inference_dense_1_layer_call_and_return_conditional_losses_56005

inputs4
!tensordot_readvariableop_resource:	 �-.
biasadd_readvariableop_resource:	�-
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �-*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� �
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������-\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�-Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������-s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�-*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������-[
SoftmaxSoftmaxBiasAdd:output:0*
T0*,
_output_shapes
:����������-�
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �-*
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*,
_output_shapes
:����������-�
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_jerry_model_layer_call_fn_55696
input_1	
unknown:	�- 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:	 �-

unknown_20:	�-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_jerry_model_layer_call_and_return_conditional_losses_55485t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name55692:%!

_user_specified_name55690:%!

_user_specified_name55688:%!

_user_specified_name55686:%!

_user_specified_name55684:%!

_user_specified_name55682:%!

_user_specified_name55680:%!

_user_specified_name55678:%!

_user_specified_name55676:%!

_user_specified_name55674:%!

_user_specified_name55672:%!

_user_specified_name55670:%
!

_user_specified_name55668:%	!

_user_specified_name55666:%!

_user_specified_name55664:%!

_user_specified_name55662:%!

_user_specified_name55660:%!

_user_specified_name55658:%!

_user_specified_name55656:%!

_user_specified_name55654:%!

_user_specified_name55652:%!

_user_specified_name55650:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�=
__inference__traced_save_56849
file_prefix>
+read_disablecopyonread_embedding_embeddings:	�- A
/read_1_disablecopyonread_embedding_1_embeddings: :
'read_2_disablecopyonread_dense_1_kernel:	 �-4
%read_3_disablecopyonread_dense_1_bias:	�-@
2read_4_disablecopyonread_batch_normalization_gamma: ?
1read_5_disablecopyonread_batch_normalization_beta: F
8read_6_disablecopyonread_batch_normalization_moving_mean: J
<read_7_disablecopyonread_batch_normalization_moving_variance: B
4read_8_disablecopyonread_batch_normalization_1_gamma: A
3read_9_disablecopyonread_batch_normalization_1_beta: I
;read_10_disablecopyonread_batch_normalization_1_moving_mean: M
?read_11_disablecopyonread_batch_normalization_1_moving_variance: A
3read_12_disablecopyonread_layer_normalization_gamma: @
2read_13_disablecopyonread_layer_normalization_beta: Q
;read_14_disablecopyonread_multi_head_attention_query_kernel: K
9read_15_disablecopyonread_multi_head_attention_query_bias:O
9read_16_disablecopyonread_multi_head_attention_key_kernel: I
7read_17_disablecopyonread_multi_head_attention_key_bias:Q
;read_18_disablecopyonread_multi_head_attention_value_kernel: K
9read_19_disablecopyonread_multi_head_attention_value_bias:\
Fread_20_disablecopyonread_multi_head_attention_attention_output_kernel: R
Dread_21_disablecopyonread_multi_head_attention_attention_output_bias: -
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: H
5read_24_disablecopyonread_adam_m_embedding_embeddings:	�- H
5read_25_disablecopyonread_adam_v_embedding_embeddings:	�- I
7read_26_disablecopyonread_adam_m_embedding_1_embeddings: I
7read_27_disablecopyonread_adam_v_embedding_1_embeddings: B
/read_28_disablecopyonread_adam_m_dense_1_kernel:	 �-B
/read_29_disablecopyonread_adam_v_dense_1_kernel:	 �-<
-read_30_disablecopyonread_adam_m_dense_1_bias:	�-<
-read_31_disablecopyonread_adam_v_dense_1_bias:	�-H
:read_32_disablecopyonread_adam_m_batch_normalization_gamma: H
:read_33_disablecopyonread_adam_v_batch_normalization_gamma: G
9read_34_disablecopyonread_adam_m_batch_normalization_beta: G
9read_35_disablecopyonread_adam_v_batch_normalization_beta: J
<read_36_disablecopyonread_adam_m_batch_normalization_1_gamma: J
<read_37_disablecopyonread_adam_v_batch_normalization_1_gamma: I
;read_38_disablecopyonread_adam_m_batch_normalization_1_beta: I
;read_39_disablecopyonread_adam_v_batch_normalization_1_beta: H
:read_40_disablecopyonread_adam_m_layer_normalization_gamma: H
:read_41_disablecopyonread_adam_v_layer_normalization_gamma: G
9read_42_disablecopyonread_adam_m_layer_normalization_beta: G
9read_43_disablecopyonread_adam_v_layer_normalization_beta: X
Bread_44_disablecopyonread_adam_m_multi_head_attention_query_kernel: X
Bread_45_disablecopyonread_adam_v_multi_head_attention_query_kernel: R
@read_46_disablecopyonread_adam_m_multi_head_attention_query_bias:R
@read_47_disablecopyonread_adam_v_multi_head_attention_query_bias:V
@read_48_disablecopyonread_adam_m_multi_head_attention_key_kernel: V
@read_49_disablecopyonread_adam_v_multi_head_attention_key_kernel: P
>read_50_disablecopyonread_adam_m_multi_head_attention_key_bias:P
>read_51_disablecopyonread_adam_v_multi_head_attention_key_bias:X
Bread_52_disablecopyonread_adam_m_multi_head_attention_value_kernel: X
Bread_53_disablecopyonread_adam_v_multi_head_attention_value_kernel: R
@read_54_disablecopyonread_adam_m_multi_head_attention_value_bias:R
@read_55_disablecopyonread_adam_v_multi_head_attention_value_bias:c
Mread_56_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel: c
Mread_57_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel: Y
Kread_58_disablecopyonread_adam_m_multi_head_attention_attention_output_bias: Y
Kread_59_disablecopyonread_adam_v_multi_head_attention_attention_output_bias: 
savev2_const
identity_121��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�- *
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�- b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�- �
Read_1/DisableCopyOnReadDisableCopyOnRead/read_1_disablecopyonread_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp/read_1_disablecopyonread_embedding_1_embeddings^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

: {
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �-*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �-d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �-y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�-*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�-`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�-�
Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_batch_normalization_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_batch_normalization_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead8read_6_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp8read_6_disablecopyonread_batch_normalization_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead<read_7_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp<read_7_disablecopyonread_batch_normalization_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead3read_12_disablecopyonread_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp3read_12_disablecopyonread_layer_normalization_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead2read_13_disablecopyonread_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp2read_13_disablecopyonread_layer_normalization_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead;read_14_disablecopyonread_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp;read_14_disablecopyonread_multi_head_attention_query_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead9read_15_disablecopyonread_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp9read_15_disablecopyonread_multi_head_attention_query_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_16/DisableCopyOnReadDisableCopyOnRead9read_16_disablecopyonread_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp9read_16_disablecopyonread_multi_head_attention_key_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead7read_17_disablecopyonread_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp7read_17_disablecopyonread_multi_head_attention_key_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_18/DisableCopyOnReadDisableCopyOnRead;read_18_disablecopyonread_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp;read_18_disablecopyonread_multi_head_attention_value_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead9read_19_disablecopyonread_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp9read_19_disablecopyonread_multi_head_attention_value_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_20/DisableCopyOnReadDisableCopyOnReadFread_20_disablecopyonread_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpFread_20_disablecopyonread_multi_head_attention_attention_output_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_21/DisableCopyOnReadDisableCopyOnReadDread_21_disablecopyonread_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpDread_21_disablecopyonread_multi_head_attention_attention_output_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead5read_24_disablecopyonread_adam_m_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp5read_24_disablecopyonread_adam_m_embedding_embeddings^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�- *
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�- f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�- �
Read_25/DisableCopyOnReadDisableCopyOnRead5read_25_disablecopyonread_adam_v_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp5read_25_disablecopyonread_adam_v_embedding_embeddings^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�- *
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�- f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�- �
Read_26/DisableCopyOnReadDisableCopyOnRead7read_26_disablecopyonread_adam_m_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp7read_26_disablecopyonread_adam_m_embedding_1_embeddings^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_27/DisableCopyOnReadDisableCopyOnRead7read_27_disablecopyonread_adam_v_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp7read_27_disablecopyonread_adam_v_embedding_1_embeddings^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_28/DisableCopyOnReadDisableCopyOnRead/read_28_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp/read_28_disablecopyonread_adam_m_dense_1_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �-*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �-f
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �-�
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_adam_v_dense_1_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �-*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �-f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �-�
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_m_dense_1_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�-*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�-b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:�-�
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_adam_v_dense_1_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�-*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�-b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:�-�
Read_32/DisableCopyOnReadDisableCopyOnRead:read_32_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp:read_32_disablecopyonread_adam_m_batch_normalization_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead:read_33_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp:read_33_disablecopyonread_adam_v_batch_normalization_gamma^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_34/DisableCopyOnReadDisableCopyOnRead9read_34_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp9read_34_disablecopyonread_adam_m_batch_normalization_beta^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_35/DisableCopyOnReadDisableCopyOnRead9read_35_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp9read_35_disablecopyonread_adam_v_batch_normalization_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_36/DisableCopyOnReadDisableCopyOnRead<read_36_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp<read_36_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_37/DisableCopyOnReadDisableCopyOnRead<read_37_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp<read_37_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_38/DisableCopyOnReadDisableCopyOnRead;read_38_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp;read_38_disablecopyonread_adam_m_batch_normalization_1_beta^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_39/DisableCopyOnReadDisableCopyOnRead;read_39_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp;read_39_disablecopyonread_adam_v_batch_normalization_1_beta^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_40/DisableCopyOnReadDisableCopyOnRead:read_40_disablecopyonread_adam_m_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp:read_40_disablecopyonread_adam_m_layer_normalization_gamma^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_41/DisableCopyOnReadDisableCopyOnRead:read_41_disablecopyonread_adam_v_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp:read_41_disablecopyonread_adam_v_layer_normalization_gamma^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead9read_42_disablecopyonread_adam_m_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp9read_42_disablecopyonread_adam_m_layer_normalization_beta^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_43/DisableCopyOnReadDisableCopyOnRead9read_43_disablecopyonread_adam_v_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp9read_43_disablecopyonread_adam_v_layer_normalization_beta^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_44/DisableCopyOnReadDisableCopyOnReadBread_44_disablecopyonread_adam_m_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpBread_44_disablecopyonread_adam_m_multi_head_attention_query_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_45/DisableCopyOnReadDisableCopyOnReadBread_45_disablecopyonread_adam_v_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpBread_45_disablecopyonread_adam_v_multi_head_attention_query_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_46/DisableCopyOnReadDisableCopyOnRead@read_46_disablecopyonread_adam_m_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp@read_46_disablecopyonread_adam_m_multi_head_attention_query_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_47/DisableCopyOnReadDisableCopyOnRead@read_47_disablecopyonread_adam_v_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp@read_47_disablecopyonread_adam_v_multi_head_attention_query_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_48/DisableCopyOnReadDisableCopyOnRead@read_48_disablecopyonread_adam_m_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp@read_48_disablecopyonread_adam_m_multi_head_attention_key_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_49/DisableCopyOnReadDisableCopyOnRead@read_49_disablecopyonread_adam_v_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp@read_49_disablecopyonread_adam_v_multi_head_attention_key_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_50/DisableCopyOnReadDisableCopyOnRead>read_50_disablecopyonread_adam_m_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp>read_50_disablecopyonread_adam_m_multi_head_attention_key_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_51/DisableCopyOnReadDisableCopyOnRead>read_51_disablecopyonread_adam_v_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp>read_51_disablecopyonread_adam_v_multi_head_attention_key_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_52/DisableCopyOnReadDisableCopyOnReadBread_52_disablecopyonread_adam_m_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpBread_52_disablecopyonread_adam_m_multi_head_attention_value_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0t
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: k
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_53/DisableCopyOnReadDisableCopyOnReadBread_53_disablecopyonread_adam_v_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpBread_53_disablecopyonread_adam_v_multi_head_attention_value_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0t
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: k
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_54/DisableCopyOnReadDisableCopyOnRead@read_54_disablecopyonread_adam_m_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp@read_54_disablecopyonread_adam_m_multi_head_attention_value_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_55/DisableCopyOnReadDisableCopyOnRead@read_55_disablecopyonread_adam_v_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp@read_55_disablecopyonread_adam_v_multi_head_attention_value_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_56/DisableCopyOnReadDisableCopyOnReadMread_56_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpMread_56_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0t
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: k
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_57/DisableCopyOnReadDisableCopyOnReadMread_57_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpMread_57_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0t
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: k
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*"
_output_shapes
: �
Read_58/DisableCopyOnReadDisableCopyOnReadKread_58_disablecopyonread_adam_m_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpKread_58_disablecopyonread_adam_m_multi_head_attention_attention_output_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_59/DisableCopyOnReadDisableCopyOnReadKread_59_disablecopyonread_adam_v_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpKread_59_disablecopyonread_adam_v_multi_head_attention_attention_output_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *K
dtypesA
?2=	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_120Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_121IdentityIdentity_120:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_121Identity_121:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:==9

_output_shapes
: 

_user_specified_nameConst:Q<M
K
_user_specified_name31Adam/v/multi_head_attention/attention_output/bias:Q;M
K
_user_specified_name31Adam/m/multi_head_attention/attention_output/bias:S:O
M
_user_specified_name53Adam/v/multi_head_attention/attention_output/kernel:S9O
M
_user_specified_name53Adam/m/multi_head_attention/attention_output/kernel:F8B
@
_user_specified_name(&Adam/v/multi_head_attention/value/bias:F7B
@
_user_specified_name(&Adam/m/multi_head_attention/value/bias:H6D
B
_user_specified_name*(Adam/v/multi_head_attention/value/kernel:H5D
B
_user_specified_name*(Adam/m/multi_head_attention/value/kernel:D4@
>
_user_specified_name&$Adam/v/multi_head_attention/key/bias:D3@
>
_user_specified_name&$Adam/m/multi_head_attention/key/bias:F2B
@
_user_specified_name(&Adam/v/multi_head_attention/key/kernel:F1B
@
_user_specified_name(&Adam/m/multi_head_attention/key/kernel:F0B
@
_user_specified_name(&Adam/v/multi_head_attention/query/bias:F/B
@
_user_specified_name(&Adam/m/multi_head_attention/query/bias:H.D
B
_user_specified_name*(Adam/v/multi_head_attention/query/kernel:H-D
B
_user_specified_name*(Adam/m/multi_head_attention/query/kernel:?,;
9
_user_specified_name!Adam/v/layer_normalization/beta:?+;
9
_user_specified_name!Adam/m/layer_normalization/beta:@*<
:
_user_specified_name" Adam/v/layer_normalization/gamma:@)<
:
_user_specified_name" Adam/m/layer_normalization/gamma:A(=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A'=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B&>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B%>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:?$;
9
_user_specified_name!Adam/v/batch_normalization/beta:?#;
9
_user_specified_name!Adam/m/batch_normalization/beta:@"<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@!<
:
_user_specified_name" Adam/m/batch_normalization/gamma:3 /
-
_user_specified_nameAdam/v/dense_1/bias:3/
-
_user_specified_nameAdam/m/dense_1/bias:51
/
_user_specified_nameAdam/v/dense_1/kernel:51
/
_user_specified_nameAdam/m/dense_1/kernel:=9
7
_user_specified_nameAdam/v/embedding_1/embeddings:=9
7
_user_specified_nameAdam/m/embedding_1/embeddings:;7
5
_user_specified_nameAdam/v/embedding/embeddings:;7
5
_user_specified_nameAdam/m/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:84
2
_user_specified_namelayer_normalization/beta:95
3
_user_specified_namelayer_normalization/gamma:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:62
0
_user_specified_nameembedding_1/embeddings:40
.
_user_specified_nameembedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_56196

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:��������� v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:��������� f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:��������� \
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56165

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�a
�
F__inference_jerry_model_layer_call_and_return_conditional_losses_55485
input_1	"
embedding_55254:	�- #
embedding_1_55269: '
batch_normalization_55273: '
batch_normalization_55275: '
batch_normalization_55277: '
batch_normalization_55279: '
layer_normalization_55304: '
layer_normalization_55306: 0
multi_head_attention_55385: ,
multi_head_attention_55387:0
multi_head_attention_55389: ,
multi_head_attention_55391:0
multi_head_attention_55393: ,
multi_head_attention_55395:0
multi_head_attention_55397: (
multi_head_attention_55399: )
batch_normalization_1_55402: )
batch_normalization_1_55404: )
batch_normalization_1_55406: )
batch_normalization_1_55408:  
dense_1_55459:	 �-
dense_1_55461:	�-
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�!dropout_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_55254*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55253M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallrange:output:0embedding_1_55269*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_55268�
addAddV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:��������� �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCalladd:z:0batch_normalization_55273batch_normalization_55275batch_normalization_55277batch_normalization_55279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55117�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0layer_normalization_55304layer_normalization_55306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_55303�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_55385multi_head_attention_55387multi_head_attention_55389multi_head_attention_55391multi_head_attention_55393multi_head_attention_55395multi_head_attention_55397multi_head_attention_55399*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55384�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0batch_normalization_1_55402batch_normalization_1_55404batch_normalization_1_55406batch_normalization_1_55408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55197�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_55422�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_55459dense_1_55461*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_55458
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_55459*
_output_shapes
:	 �-*
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55385*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55389*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55393*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmulti_head_attention_55397*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCallO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:%!

_user_specified_name55461:%!

_user_specified_name55459:%!

_user_specified_name55408:%!

_user_specified_name55406:%!

_user_specified_name55404:%!

_user_specified_name55402:%!

_user_specified_name55399:%!

_user_specified_name55397:%!

_user_specified_name55395:%!

_user_specified_name55393:%!

_user_specified_name55391:%!

_user_specified_name55389:%
!

_user_specified_name55387:%	!

_user_specified_name55385:%!

_user_specified_name55306:%!

_user_specified_name55304:%!

_user_specified_name55279:%!

_user_specified_name55277:%!

_user_specified_name55275:%!

_user_specified_name55273:%!

_user_specified_name55269:%!

_user_specified_name55254:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
4__inference_multi_head_attention_layer_call_fn_56267	
query	
value
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55588s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%	!

_user_specified_name56263:%!

_user_specified_name56261:%!

_user_specified_name56259:%!

_user_specified_name56257:%!

_user_specified_name56255:%!

_user_specified_name56253:%!

_user_specified_name56251:%!

_user_specified_name56249:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_55619

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:��������� _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�&
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56065

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_56218

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:��������� e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
4__inference_multi_head_attention_layer_call_fn_56245	
query	
value
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55384s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%	!

_user_specified_name56241:%!

_user_specified_name56239:%!

_user_specified_name56237:%!

_user_specified_name56235:%!

_user_specified_name56233:%!

_user_specified_name56231:%!

_user_specified_name56229:%!

_user_specified_name56227:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�	
�
5__inference_batch_normalization_1_layer_call_fn_56111

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55217|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name56107:%!

_user_specified_name56105:%!

_user_specified_name56103:%!

_user_specified_name56101:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�a
�	
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56419	
query	
valueA
+query_einsum_einsum_readvariableop_resource: 3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource: 1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource: 3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource: :
,attention_output_add_readvariableop_resource: 
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOpH
ShapeShapequery*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
Shape_1Shapevalue*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
ones/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
ones/packedPackones/packed/0:output:0strided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zd
onesFillones/packed:output:0ones/Const:output:0*
T0
*"
_output_shapes
:c
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������Z
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
MatrixBandPartMatrixBandPartones:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
Tindex0*
T0
*"
_output_shapes
:�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:����������
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbeY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������y

ExpandDims
ExpandDimsMatrixBandPart:band:0ExpandDims/dim:output:0*
T0
*&
_output_shapes
:i
softmax/CastCastExpandDims:output:0*

DstT0*

SrcT0
*&
_output_shapes
:R
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*&
_output_shapes
:R
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn�l
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*&
_output_shapes
:w
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*/
_output_shapes
:���������e
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:��������� *
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOpO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�a
�	
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55588	
query	
valueA
+query_einsum_einsum_readvariableop_resource: 3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource: 1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource: 3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource: :
,attention_output_add_readvariableop_resource: 
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOpH
ShapeShapequery*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
Shape_1Shapevalue*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
ones/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
ones/packedPackones/packed/0:output:0strided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zd
onesFillones/packed:output:0ones/Const:output:0*
T0
*"
_output_shapes
:c
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������Z
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
MatrixBandPartMatrixBandPartones:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
Tindex0*
T0
*"
_output_shapes
:�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:����������
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbeY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������y

ExpandDims
ExpandDimsMatrixBandPart:band:0ExpandDims/dim:output:0*
T0
*&
_output_shapes
:i
softmax/CastCastExpandDims:output:0*

DstT0*

SrcT0
*&
_output_shapes
:R
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*&
_output_shapes
:R
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn�l
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*&
_output_shapes
:w
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*/
_output_shapes
:���������e
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:��������� *
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOpO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�	
�
3__inference_batch_normalization_layer_call_fn_56031

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55137|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name56027:%!

_user_specified_name56025:%!

_user_specified_name56023:%!

_user_specified_name56021:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_1_layer_call_fn_56098

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55197|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name56094:%!

_user_specified_name56092:%!

_user_specified_name56090:%!

_user_specified_name56088:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�a
�	
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_55384	
query	
valueA
+query_einsum_einsum_readvariableop_resource: 3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource: 1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource: 3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource: :
,attention_output_add_readvariableop_resource: 
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp�Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp�Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOpH
ShapeShapequery*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
Shape_1Shapevalue*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
ones/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
ones/packedPackones/packed/0:output:0strided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zd
onesFillones/packed:output:0ones/Const:output:0*
T0
*"
_output_shapes
:c
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������Z
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
MatrixBandPartMatrixBandPartones:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
Tindex0*
T0
*"
_output_shapes
:�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:����������
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbeY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������y

ExpandDims
ExpandDimsMatrixBandPart:band:0ExpandDims/dim:output:0*
T0
*&
_output_shapes
:i
softmax/CastCastExpandDims:output:0*

DstT0*

SrcT0
*&
_output_shapes
:R
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*&
_output_shapes
:R
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn�l
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*&
_output_shapes
:w
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*/
_output_shapes
:���������e
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:��������� *
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/query/kernel/Regularizer/L2LossL2LossKmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/query/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/query/kernel/Regularizer/mulMul<multi_head_attention/query/kernel/Regularizer/mul/x:output:0=multi_head_attention/query/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
2multi_head_attention/key/kernel/Regularizer/L2LossL2LossImulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: v
1multi_head_attention/key/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/multi_head_attention/key/kernel/Regularizer/mulMul:multi_head_attention/key/kernel/Regularizer/mul/x:output:0;multi_head_attention/key/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
4multi_head_attention/value/kernel/Regularizer/L2LossL2LossKmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: x
3multi_head_attention/value/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1multi_head_attention/value/kernel/Regularizer/mulMul<multi_head_attention/value/kernel/Regularizer/mul/x:output:0=multi_head_attention/value/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0�
?multi_head_attention/attention_output/kernel/Regularizer/L2LossL2LossVmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: �
>multi_head_attention/attention_output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<multi_head_attention/attention_output/kernel/Regularizer/mulMulGmulti_head_attention/attention_output/kernel/Regularizer/mul/x:output:0Hmulti_head_attention/attention_output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOpO^multi_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpB^multi_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpD^multi_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:��������� :��������� : : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp2�
Nmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOpNmulti_head_attention/attention_output/kernel/Regularizer/L2Loss/ReadVariableOp2�
Amulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOpAmulti_head_attention/key/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/query/kernel/Regularizer/L2Loss/ReadVariableOp2�
Cmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOpCmulti_head_attention/value/kernel/Regularizer/L2Loss/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:��������� 

_user_specified_namevalue:R N
+
_output_shapes
:��������� 

_user_specified_namequery
�
b
)__inference_dropout_1_layer_call_fn_56201

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_55422s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_55961

inputs(
embedding_lookup_55956: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55956inputs*
Tindices0*)
_class
loc:@embedding_lookup/55956*
_output_shapes

: *
dtype0i
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*
_output_shapes

: h
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*
_output_shapes

: 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:: 2$
embedding_lookupembedding_lookup:%!

_user_specified_name55956:B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_55137

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_55422

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:��������� e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0	���������A
output_15
StatefulPartitionedCall:0����������-tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
token_embedding
	pos_embedding


dense1

dense2
	norm1
	norm2

layernorm1

layernorm2
dropout1
dropout2
	multihead
lstm
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18
)19
*20
+21"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
"8
#9
$10
%11
&12
'13
(14
)15
*16
+17"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
2trace_0
3trace_12�
+__inference_jerry_model_layer_call_fn_55696
+__inference_jerry_model_layer_call_fn_55745�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z2trace_0z3trace_1
�
4trace_0
5trace_12�
F__inference_jerry_model_layer_call_and_return_conditional_losses_55485
F__inference_jerry_model_layer_call_and_return_conditional_losses_55647�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z4trace_0z5trace_1
�B�
 __inference__wrapped_model_55083input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
(
B	keras_api"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
Vaxis
	gamma
beta
 moving_mean
!moving_variance"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	"gamma
#beta"
_tf_keras_layer
(
^	keras_api"
_tf_keras_layer
?
_	keras_api
`_random_generator"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
n_query_dense
o
_key_dense
p_value_dense
q_softmax
r_dropout_layer
s_output_dense"
_tf_keras_layer
]
t	keras_api
u_random_generator
vcell
w
state_spec"
_tf_keras_rnn_layer
�
x
_variables
y_iterations
z_learning_rate
{_index_dict
|
_momentums
}_velocities
~_update_step_xla"
experimentalOptimizer
,
serving_default"
signature_map
':%	�- 2embedding/embeddings
(:& 2embedding_1/embeddings
!:	 �-2dense_1/kernel
:�-2dense_1/bias
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
':% 2layer_normalization/gamma
&:$ 2layer_normalization/beta
7:5 2!multi_head_attention/query/kernel
1:/2multi_head_attention/query/bias
5:3 2multi_head_attention/key/kernel
/:-2multi_head_attention/key/bias
7:5 2!multi_head_attention/value/kernel
1:/2multi_head_attention/value/bias
B:@ 2,multi_head_attention/attention_output/kernel
8:6 2*multi_head_attention/attention_output/bias
�
�trace_02�
__inference_loss_fn_0_55931�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
<
0
1
 2
!3"
trackable_list_wrapper
v
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_jerry_model_layer_call_fn_55696input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_jerry_model_layer_call_fn_55745input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_jerry_model_layer_call_and_return_conditional_losses_55485input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_jerry_model_layer_call_and_return_conditional_losses_55647input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_embedding_layer_call_fn_55938�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_embedding_layer_call_and_return_conditional_losses_55946�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_embedding_1_layer_call_fn_55953�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_embedding_1_layer_call_and_return_conditional_losses_55961�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_55970�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_56005�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_batch_normalization_layer_call_fn_56018
3__inference_batch_normalization_layer_call_fn_56031�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56065
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56085�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_1_layer_call_fn_56098
5__inference_batch_normalization_1_layer_call_fn_56111�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56145
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56165�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_layer_normalization_layer_call_fn_56174�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_56196�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_1_layer_call_fn_56201
)__inference_dropout_1_layer_call_fn_56206�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_1_layer_call_and_return_conditional_losses_56218
D__inference_dropout_1_layer_call_and_return_conditional_losses_56223�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_multi_head_attention_layer_call_fn_56245
4__inference_multi_head_attention_layer_call_fn_56267�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56343
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56419�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

$kernel
%bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

&kernel
'bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

(kernel
)bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

*kernel
+bias"
_tf_keras_layer
"
_generic_user_object
"
_generic_user_object
R
�	keras_api
�_random_generator
�
state_size"
_tf_keras_layer
 "
trackable_list_wrapper
�
y0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_55903input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_55931"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_embedding_layer_call_fn_55938inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_embedding_layer_call_and_return_conditional_losses_55946inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_embedding_1_layer_call_fn_55953inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_embedding_1_layer_call_and_return_conditional_losses_55961inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_55970inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_56005inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_batch_normalization_layer_call_fn_56018inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_batch_normalization_layer_call_fn_56031inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56065inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56085inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_1_layer_call_fn_56098inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_1_layer_call_fn_56111inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56145inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56165inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_layer_normalization_layer_call_fn_56174inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_56196inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_1_layer_call_fn_56201inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_1_layer_call_fn_56206inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_56218inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_56223inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�trace_02�
__inference_loss_fn_1_56443�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_56451�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_56459�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_56467�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
J
n0
o1
p2
q3
r4
s5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_multi_head_attention_layer_call_fn_56245queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_multi_head_attention_layer_call_fn_56267queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56343queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56419queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
,:*	�- 2Adam/m/embedding/embeddings
,:*	�- 2Adam/v/embedding/embeddings
-:+ 2Adam/m/embedding_1/embeddings
-:+ 2Adam/v/embedding_1/embeddings
&:$	 �-2Adam/m/dense_1/kernel
&:$	 �-2Adam/v/dense_1/kernel
 :�-2Adam/m/dense_1/bias
 :�-2Adam/v/dense_1/bias
,:* 2 Adam/m/batch_normalization/gamma
,:* 2 Adam/v/batch_normalization/gamma
+:) 2Adam/m/batch_normalization/beta
+:) 2Adam/v/batch_normalization/beta
.:, 2"Adam/m/batch_normalization_1/gamma
.:, 2"Adam/v/batch_normalization_1/gamma
-:+ 2!Adam/m/batch_normalization_1/beta
-:+ 2!Adam/v/batch_normalization_1/beta
,:* 2 Adam/m/layer_normalization/gamma
,:* 2 Adam/v/layer_normalization/gamma
+:) 2Adam/m/layer_normalization/beta
+:) 2Adam/v/layer_normalization/beta
<:: 2(Adam/m/multi_head_attention/query/kernel
<:: 2(Adam/v/multi_head_attention/query/kernel
6:42&Adam/m/multi_head_attention/query/bias
6:42&Adam/v/multi_head_attention/query/bias
::8 2&Adam/m/multi_head_attention/key/kernel
::8 2&Adam/v/multi_head_attention/key/kernel
4:22$Adam/m/multi_head_attention/key/bias
4:22$Adam/v/multi_head_attention/key/bias
<:: 2(Adam/m/multi_head_attention/value/kernel
<:: 2(Adam/v/multi_head_attention/value/kernel
6:42&Adam/m/multi_head_attention/value/bias
6:42&Adam/v/multi_head_attention/value/bias
G:E 23Adam/m/multi_head_attention/attention_output/kernel
G:E 23Adam/v/multi_head_attention/attention_output/kernel
=:; 21Adam/m/multi_head_attention/attention_output/bias
=:; 21Adam/v/multi_head_attention/attention_output/bias
�B�
__inference_loss_fn_1_56443"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_56451"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_56459"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_56467"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_55083�"#$%&'()*+! 0�-
&�#
!�
input_1���������	
� "8�5
3
output_1'�$
output_1����������-�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56145� !D�A
:�7
-�*
inputs������������������ 
p

 
� "9�6
/�,
tensor_0������������������ 
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_56165�! D�A
:�7
-�*
inputs������������������ 
p 

 
� "9�6
/�,
tensor_0������������������ 
� �
5__inference_batch_normalization_1_layer_call_fn_56098| !D�A
:�7
-�*
inputs������������������ 
p

 
� ".�+
unknown������������������ �
5__inference_batch_normalization_1_layer_call_fn_56111|! D�A
:�7
-�*
inputs������������������ 
p 

 
� ".�+
unknown������������������ �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56065�D�A
:�7
-�*
inputs������������������ 
p

 
� "9�6
/�,
tensor_0������������������ 
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_56085�D�A
:�7
-�*
inputs������������������ 
p 

 
� "9�6
/�,
tensor_0������������������ 
� �
3__inference_batch_normalization_layer_call_fn_56018|D�A
:�7
-�*
inputs������������������ 
p

 
� ".�+
unknown������������������ �
3__inference_batch_normalization_layer_call_fn_56031|D�A
:�7
-�*
inputs������������������ 
p 

 
� ".�+
unknown������������������ �
B__inference_dense_1_layer_call_and_return_conditional_losses_56005l3�0
)�&
$�!
inputs��������� 
� "1�.
'�$
tensor_0����������-
� �
'__inference_dense_1_layer_call_fn_55970a3�0
)�&
$�!
inputs��������� 
� "&�#
unknown����������-�
D__inference_dropout_1_layer_call_and_return_conditional_losses_56218k7�4
-�*
$�!
inputs��������� 
p
� "0�-
&�#
tensor_0��������� 
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_56223k7�4
-�*
$�!
inputs��������� 
p 
� "0�-
&�#
tensor_0��������� 
� �
)__inference_dropout_1_layer_call_fn_56201`7�4
-�*
$�!
inputs��������� 
p
� "%�"
unknown��������� �
)__inference_dropout_1_layer_call_fn_56206`7�4
-�*
$�!
inputs��������� 
p 
� "%�"
unknown��������� �
F__inference_embedding_1_layer_call_and_return_conditional_losses_55961L"�
�
�
inputs
� "#� 
�
tensor_0 
� p
+__inference_embedding_1_layer_call_fn_55953A"�
�
�
inputs
� "�
unknown �
D__inference_embedding_layer_call_and_return_conditional_losses_55946f/�,
%�"
 �
inputs���������	
� "0�-
&�#
tensor_0��������� 
� �
)__inference_embedding_layer_call_fn_55938[/�,
%�"
 �
inputs���������	
� "%�"
unknown��������� �
F__inference_jerry_model_layer_call_and_return_conditional_losses_55485�"#$%&'()*+ !4�1
*�'
!�
input_1���������	
p
� "1�.
'�$
tensor_0����������-
� �
F__inference_jerry_model_layer_call_and_return_conditional_losses_55647�"#$%&'()*+! 4�1
*�'
!�
input_1���������	
p 
� "1�.
'�$
tensor_0����������-
� �
+__inference_jerry_model_layer_call_fn_55696v"#$%&'()*+ !4�1
*�'
!�
input_1���������	
p
� "&�#
unknown����������-�
+__inference_jerry_model_layer_call_fn_55745v"#$%&'()*+! 4�1
*�'
!�
input_1���������	
p 
� "&�#
unknown����������-�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_56196k"#3�0
)�&
$�!
inputs��������� 
� "0�-
&�#
tensor_0��������� 
� �
3__inference_layer_normalization_layer_call_fn_56174`"#3�0
)�&
$�!
inputs��������� 
� "%�"
unknown��������� C
__inference_loss_fn_0_55931$�

� 
� "�
unknown C
__inference_loss_fn_1_56443$$�

� 
� "�
unknown C
__inference_loss_fn_2_56451$&�

� 
� "�
unknown C
__inference_loss_fn_3_56459$(�

� 
� "�
unknown C
__inference_loss_fn_4_56467$*�

� 
� "�
unknown �
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56343�$%&'()*+k�h
a�^
#� 
query��������� 
#� 
value��������� 

 

 
p 
p
p
� "0�-
&�#
tensor_0��������� 
� �
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_56419�$%&'()*+k�h
a�^
#� 
query��������� 
#� 
value��������� 

 

 
p 
p 
p
� "0�-
&�#
tensor_0��������� 
� �
4__inference_multi_head_attention_layer_call_fn_56245�$%&'()*+k�h
a�^
#� 
query��������� 
#� 
value��������� 

 

 
p 
p
p
� "%�"
unknown��������� �
4__inference_multi_head_attention_layer_call_fn_56267�$%&'()*+k�h
a�^
#� 
query��������� 
#� 
value��������� 

 

 
p 
p 
p
� "%�"
unknown��������� �
#__inference_signature_wrapper_55903�"#$%&'()*+! ;�8
� 
1�.
,
input_1!�
input_1���������	"8�5
3
output_1'�$
output_1����������-