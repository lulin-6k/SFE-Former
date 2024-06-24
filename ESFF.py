class DA(nn.Module):

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2, stride=1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W])
        k = k.reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W])
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W])
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class ESFF(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5, dilation=[1, 3], kernel_size=3):
        super().__init__()
        head_dim = int(dim/num_heads)
        self.dim = dim
        self.lheads = int(num_heads * alpha)
        self.ldim = self.lheads * head_dim
        self.hheads = num_heads - self.lheads
        self.kernel_size = kernel_size
        self.hdim = self.hheads * head_dim
        self.num_dilation = len(1)
        self.dilation = dilation
        self.dilate_attention = nn.ModuleList(
            [DA(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.ws = window_size
        self.scale = round(qk_scale or head_dim ** -0.5, 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.lheads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.lq = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.lkv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.lproj = nn.Linear(self.l_dim, self.l_dim)
        self.classifiers = nn.Linear(75264, 2)

    def LAE_Attention(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # (B,H,W,C)-->(B,C,H,W)
        qkv = self.qkv(x)
        qkv = qkv.detach().reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def GAE_Attention(self, x):
        B, H, W, C = x.shape
        q = self.lq(x).reshape(B, H * W, self.lheads, self.ldim // self.lheads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x = x.permute(0, 3, 1, 2)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.lkv(x).reshape(B, -1, 2, self.lheads, self.ldim // self.lheads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.lkv(x).reshape(B, -1, 2, self.lheads, self.ldim // self.lheads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.ldim)
        x = self.lproj(x)
        return x


    def forward(self, x):
        B, H, W, C = x.shape
        x= x.reshape(4,7,H,W,C)
        x = x.mean(dim=1)

        LAE_out = self.LAE_Attention(x)
        GAE_out = self.GAE_Attention(x)
        b1,h1,w1,c1 = LAE_out.shape
        b2,h2,w2,c2 = GAE_out.shape

        x = torch.cat((LAE_out, GAE_out), dim=-1)
        x = x.reshape(4,H,W,c1+c2)
        x = x.view(x.size(0), -1)
        x = self.classifiers(x)

        return x
