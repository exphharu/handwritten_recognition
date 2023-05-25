import numpy as np

# REF: github.com/skillup-ai/tettei-engineer/blob/main/chapter13/Q2_ans.py
# REF: tatsukioike.com/pythoncnn/0057/

def im2col(input_data, filter_h, filter_w, stride, pad, constant_values=0):
    """
    input_data : (データ数, チャンネル数, 高さ, 幅)の4次元配列
    filter_h : フィルタの高さ
    filter_w : フィルタの幅
    stride : ストライドサイズ
    pad : パディングサイズ
    constant_values : パディング処理で埋める際の値
    """

    # 入力データのデータ数, チャンネル数, 高さ, 幅を取得
    N, C, H, W = input_data.shape

    # 出力データの高さ(端数は切り捨てる)
    out_h = (H + 2 * pad - filter_h) // stride + 1

    # 出力データの幅(端数は切り捨てる)
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # パディング処理
    img = np.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        "constant",
        constant_values=constant_values,
    )

    # 配列の初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # フィルタ内のある1要素に対応する画像中の画素を取り出してcolに代入
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 軸を入れ替えて、2次元配列(行列)に変換する
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col : 2次元配列
    input_shape : 戻すデータの形状
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    return : (データ数, チャンネル数, 高さ, 幅)の4次元配列. 画像データの形式を想定している
    -------
    """
    
    # 入力画像(元画像)のデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_shape
    
    # 出力データの高さ(端数は切り捨てる)
    out_h = (H + 2*pad - filter_h)//stride + 1 

    # 出力データの幅(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 
    
    # colを6次元配列に
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    #画像の初期化
    #stride/paddingを考慮する
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w            
            # col 2 image
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    #周りのpadding部分は除く        
    return img[:, :, pad:H + pad, pad:W + pad]