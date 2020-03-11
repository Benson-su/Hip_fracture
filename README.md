# Hip_fracture



Image_dist: 
          測量不同group影像間的差異性, 以cosin/ssim/psnr三種方法測試

data_preprocessing: 
          將標記的json檔轉成csv並做好label的分類

training: 
          Model training, 將參數分出來成config檔，可以直接從裡面調整部分參數

models: 
          將現有的model包成一包factory, 可以統一輸入size, 方便在training與inference時提取

inference: 
          將model的結果提取特徵做Xray的heatmap分析
