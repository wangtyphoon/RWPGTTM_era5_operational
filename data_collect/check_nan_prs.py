import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def check_nan_in_npz_files(npz_path):
    """
    遍歷prs中的npz檔案，檢查每個變數是否包含NaN值，
    並建立DataFrame記錄結果
    """
    
    # 獲取所有npz檔案
    npz_files = sorted([f for f in os.listdir(npz_path) if f.endswith('.npz')])
    
    print(f"找到 {len(npz_files)} 個npz檔案")
    
    # 儲存結果的列表
    results = []
    
    # 用於記錄所有變數名稱
    all_variables = set()
    
    # 遍歷每個npz檔案
    for npz_file in tqdm(npz_files, desc="檢查npz檔案"):
        file_path = os.path.join(npz_path, npz_file)
        
        try:
            # 載入npz檔案
            data = np.load(file_path)
            
            # 檢查每個變數
            for var_name in data.keys():
                all_variables.add(var_name)
                var_data = data[var_name]
                
                # 檢查是否有NaN值
                has_nan = np.isnan(var_data).any()
                
                # 計算NaN值的數量
                nan_count = np.isnan(var_data).sum()
                
                # 計算總元素數量
                total_elements = var_data.size
                
                # 記錄結果
                results.append({
                    'file': npz_file,
                    'variable': var_name,
                    'has_nan': has_nan,
                    'nan_count': nan_count,
                    'total_elements': total_elements,
                    'nan_percentage': (nan_count / total_elements) * 100 if total_elements > 0 else 0
                })
            
            data.close()
            
        except Exception as e:
            print(f"處理檔案 {npz_file} 時發生錯誤: {e}")
            continue
    
    # 建立DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n檢查完成!")
    print(f"總共檢查了 {len(npz_files)} 個檔案")
    print(f"總共檢查了 {len(all_variables)} 個不同的變數")
    print(f"總共檢查了 {len(df)} 個變數實例")
    
    # 顯示統計資訊
    print(f"\n=== NaN檢驗統計 ===")
    total_with_nan = df['has_nan'].sum()
    total_without_nan = len(df) - total_with_nan
    
    print(f"包含NaN值的變數實例: {total_with_nan}")
    print(f"不包含NaN值的變數實例: {total_without_nan}")
    print(f"包含NaN值的比例: {(total_with_nan / len(df)) * 100:.2f}%")
    
    # 顯示每個變數的NaN統計
    print(f"\n=== 各變數NaN統計 ===")
    var_stats = df.groupby('variable').agg({
        'has_nan': ['count', 'sum'],
        'nan_count': 'sum'
    }).round(2)
    
    var_stats.columns = ['total_files', 'files_with_nan', 'total_nan_count']
    var_stats['nan_rate'] = (var_stats['files_with_nan'] / var_stats['total_files'] * 100).round(2)
    var_stats = var_stats.sort_values('nan_rate', ascending=False)
    
    print(var_stats.head(20))
    
    # # 儲存完整結果
    # output_file = 'sfcregular_nan_check_results.csv'
    # df.to_csv(output_file, index=False, encoding='utf-8-sig')
    # print(f"\n完整結果已儲存至: {output_file}")
    
    # # 儲存變數統計
    # var_stats_file = 'sfcregular_variable_nan_stats.csv'
    # var_stats.to_csv(var_stats_file, encoding='utf-8-sig')
    # print(f"變數統計已儲存至: {var_stats_file}")
    
    # 顯示有問題的檔案（如果有的話）
    if total_with_nan > 0:
        print(f"\n=== 包含NaN值的檔案範例 ===")
        problem_files = df[df['has_nan'] == True]
        print(problem_files[['file', 'variable', 'nan_count', 'nan_percentage']].head(10))
    
    return df, var_stats

if __name__ == "__main__":
    # npz檔案路徑
    npz_path = 'train/sfc/regular'
    df_results, var_statistics = check_nan_in_npz_files(npz_path)
    npz_path = 'train/sfc/average'
    df_results, var_statistics = check_nan_in_npz_files(npz_path)
    npz_path = 'train//time'
    df_results, var_statistics = check_nan_in_npz_files(npz_path)
    npz_path = 'train/prs'
    df_results, var_statistics = check_nan_in_npz_files(npz_path)
   
