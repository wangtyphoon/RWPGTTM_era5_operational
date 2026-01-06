"""
處理所有檔案的 NaN 值填補程式

針對 SST 和 SWH 進行 NaN 值填補：
- SST: NaN → 270.0
- SWH: NaN → 0.0
"""

import numpy as np
import os
import glob
from datetime import datetime

def process_all_files(directory_path=None):
    """處理所有檔案的 NaN 值填補"""
    
    directory = directory_path or "sfc/regular"
    if not os.path.exists(directory):
        print(f"錯誤: 找不到目錄 {directory}")
        return
    
    all_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    
    if not all_files:
        print(f"錯誤: 在 {directory} 中找不到 .npz 檔案")
        return
    
    print("處理所有檔案的 NaN 值填補")
    print("=" * 60)
    print(f"總檔案數量: {len(all_files)}")
    print("填補規則:")
    print("  SST: NaN → 270.0")
    print("  SWH: NaN → 0.0")
    print("=" * 60)
    
    # 統計資料
    stats = {
        'files_processed': 0,
        'files_with_sst_nan': 0,
        'files_with_swh_nan': 0,
        'total_sst_filled': 0,
        'total_swh_filled': 0,
        'start_time': datetime.now()
    }
    
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        
        # 每處理100個檔案顯示進度
        if i % 100 == 0:
            elapsed = datetime.now() - stats['start_time']
            print(f"[{i+1}/{len(all_files)}] {filename} (已處理時間: {elapsed})")
        
        try:
            # 載入數據
            data = np.load(file_path)
            new_data = {}
            
            sst_filled = 0
            swh_filled = 0
            
            for var_name in data.files:
                var_data = data[var_name].copy()
                original_nan_count = np.sum(np.isnan(var_data))
                
                if var_name == 'sst' and original_nan_count > 0:
                    var_data[np.isnan(var_data)] = 270.0
                    sst_filled = original_nan_count
                    stats['files_with_sst_nan'] += 1
                elif var_name == 'swh' and original_nan_count > 0:
                    var_data[np.isnan(var_data)] = 0.0
                    swh_filled = original_nan_count
                    stats['files_with_swh_nan'] += 1
                
                new_data[var_name] = var_data
            
            # 儲存結果
            np.savez_compressed(file_path, **new_data)
            
            stats['files_processed'] += 1
            stats['total_sst_filled'] += sst_filled
            stats['total_swh_filled'] += swh_filled
            
        except Exception as e:
            print(f"  錯誤處理 {filename}: {e}")
    
    stats['end_time'] = datetime.now()
    stats['duration'] = stats['end_time'] - stats['start_time']
    
    # 顯示統計結果
    print("\n" + "=" * 60)
    print("處理完成統計")
    print("=" * 60)
    print(f"總處理檔案: {stats['files_processed']:,}")
    print(f"有 SST NaN 的檔案: {stats['files_with_sst_nan']:,}")
    print(f"有 SWH NaN 的檔案: {stats['files_with_swh_nan']:,}")
    print(f"總 SST NaN 填補: {stats['total_sst_filled']:,}")
    print(f"總 SWH NaN 填補: {stats['total_swh_filled']:,}")
    print(f"處理時間: {stats['duration']}")
    print(f"平均每檔案處理時間: {stats['duration'] / stats['files_processed']}")
    print("=" * 60)
    
    return stats

def verify_random_files(sample_size=50):
    """驗證隨機檔案的 NaN 填補結果"""
    
    directory = "sfc/regular"
    all_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    
    # 隨機選取檔案進行驗證
    import random
    random.seed(42)  # 固定種子以便重現
    sample_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    print(f"驗證隨機 {len(sample_files)} 個檔案的 NaN 填補結果")
    print("=" * 50)
    
    verification_stats = {
        'files_checked': 0,
        'files_with_remaining_nan': 0,
        'total_remaining_sst_nan': 0,
        'total_remaining_swh_nan': 0
    }
    
    for i, file_path in enumerate(sample_files):
        filename = os.path.basename(file_path)
        
        try:
            data = np.load(file_path)
            file_has_nan = False
            
            for var_name in data.files:
                var_data = data[var_name]
                nan_count = np.sum(np.isnan(var_data))
                
                if nan_count > 0:
                    if not file_has_nan:
                        print(f"\n{filename}:")
                        file_has_nan = True
                        verification_stats['files_with_remaining_nan'] += 1
                    
                    print(f"  {var_name}: {nan_count} 個 NaN 值")
                    
                    if var_name == 'sst':
                        verification_stats['total_remaining_sst_nan'] += nan_count
                    elif var_name == 'swh':
                        verification_stats['total_remaining_swh_nan'] += nan_count
            
            verification_stats['files_checked'] += 1
            
        except Exception as e:
            print(f"錯誤檢查 {filename}: {e}")
    
    print("\n" + "=" * 50)
    print("驗證結果統計")
    print("=" * 50)
    print(f"檢查檔案數: {verification_stats['files_checked']}")
    print(f"仍有 NaN 的檔案: {verification_stats['files_with_remaining_nan']}")
    print(f"剩餘 SST NaN: {verification_stats['total_remaining_sst_nan']}")
    print(f"剩餘 SWH NaN: {verification_stats['total_remaining_swh_nan']}")
    
    if verification_stats['files_with_remaining_nan'] == 0:
        print("\n✓ 所有檢查的檔案都沒有 NaN 值！")
    else:
        print(f"\n⚠ 仍有 {verification_stats['files_with_remaining_nan']} 個檔案含有 NaN 值")
    
    print("=" * 50)
    
    return verification_stats

if __name__ == "__main__":
    print("批次處理所有檔案的 NaN 值填補")
    print("=" * 60)
    
    # 處理所有檔案
    print("開始處理所有檔案...")
    processing_stats = process_all_files()
    
    # 驗證結果
    print("\n開始驗證處理結果...")
    verification_stats = verify_random_files(sample_size=50)
    
    print(f"\n🎉 批次處理完成！")
    print(f"✅ 總共處理了 {processing_stats['files_processed']:,} 個檔案")
    print(f"✅ 填補了 {processing_stats['total_sst_filled']:,} 個 SST NaN 值")
    print(f"✅ 填補了 {processing_stats['total_swh_filled']:,} 個 SWH NaN 值")
    if verification_stats['files_with_remaining_nan'] == 0:
        print(f"✅ 驗證通過：隨機檢查的 {verification_stats['files_checked']} 個檔案都沒有 NaN 值")
    else:
        print(f"⚠️  警告：仍有 {verification_stats['files_with_remaining_nan']} 個檔案含有 NaN 值")
"""
處理所有檔案的 NaN 值填補程式

針對 SST 和 SWH 進行 NaN 值填補：
- SST: NaN → 270.0
- SWH: NaN → 0.0
"""

import numpy as np
import os
import glob
from datetime import datetime

def process_all_files(directory_path=None):
    """處理所有檔案的 NaN 值填補"""
    
    directory = directory_path or "sfc/regular"
    if not os.path.exists(directory):
        print(f"錯誤: 找不到目錄 {directory}")
        return
    
    all_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    
    if not all_files:
        print(f"錯誤: 在 {directory} 中找不到 .npz 檔案")
        return
    
    print("處理所有檔案的 NaN 值填補")
    print("=" * 60)
    print(f"總檔案數量: {len(all_files)}")
    print("填補規則:")
    print("  SST: NaN → 270.0")
    print("  SWH: NaN → 0.0")
    print("=" * 60)
    
    # 統計資料
    stats = {
        'files_processed': 0,
        'files_with_sst_nan': 0,
        'files_with_swh_nan': 0,
        'total_sst_filled': 0,
        'total_swh_filled': 0,
        'start_time': datetime.now()
    }
    
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        
        # 每處理100個檔案顯示進度
        if i % 100 == 0:
            elapsed = datetime.now() - stats['start_time']
            print(f"[{i+1}/{len(all_files)}] {filename} (已處理時間: {elapsed})")
        
        try:
            # 載入數據
            data = np.load(file_path)
            new_data = {}
            
            sst_filled = 0
            swh_filled = 0
            
            for var_name in data.files:
                var_data = data[var_name].copy()
                original_nan_count = np.sum(np.isnan(var_data))
                
                if var_name == 'sst' and original_nan_count > 0:
                    var_data[np.isnan(var_data)] = 270.0
                    sst_filled = original_nan_count
                    stats['files_with_sst_nan'] += 1
                elif var_name == 'swh' and original_nan_count > 0:
                    var_data[np.isnan(var_data)] = 0.0
                    swh_filled = original_nan_count
                    stats['files_with_swh_nan'] += 1
                
                new_data[var_name] = var_data
            
            # 儲存結果
            np.savez_compressed(file_path, **new_data)
            
            stats['files_processed'] += 1
            stats['total_sst_filled'] += sst_filled
            stats['total_swh_filled'] += swh_filled
            
        except Exception as e:
            print(f"  錯誤處理 {filename}: {e}")
    
    stats['end_time'] = datetime.now()
    stats['duration'] = stats['end_time'] - stats['start_time']
    
    # 顯示統計結果
    print("\n" + "=" * 60)
    print("處理完成統計")
    print("=" * 60)
    print(f"總處理檔案: {stats['files_processed']:,}")
    print(f"有 SST NaN 的檔案: {stats['files_with_sst_nan']:,}")
    print(f"有 SWH NaN 的檔案: {stats['files_with_swh_nan']:,}")
    print(f"總 SST NaN 填補: {stats['total_sst_filled']:,}")
    print(f"總 SWH NaN 填補: {stats['total_swh_filled']:,}")
    print(f"處理時間: {stats['duration']}")
    print(f"平均每檔案處理時間: {stats['duration'] / stats['files_processed']}")
    print("=" * 60)
    
    return stats

def verify_random_files(sample_size=50):
    """驗證隨機檔案的 NaN 填補結果"""
    
    directory = "sfc/regular"
    all_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    
    # 隨機選取檔案進行驗證
    import random
    random.seed(42)  # 固定種子以便重現
    sample_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    print(f"驗證隨機 {len(sample_files)} 個檔案的 NaN 填補結果")
    print("=" * 50)
    
    verification_stats = {
        'files_checked': 0,
        'files_with_remaining_nan': 0,
        'total_remaining_sst_nan': 0,
        'total_remaining_swh_nan': 0
    }
    
    for i, file_path in enumerate(sample_files):
        filename = os.path.basename(file_path)
        
        try:
            data = np.load(file_path)
            file_has_nan = False
            
            for var_name in data.files:
                var_data = data[var_name]
                nan_count = np.sum(np.isnan(var_data))
                
                if nan_count > 0:
                    if not file_has_nan:
                        print(f"\n{filename}:")
                        file_has_nan = True
                        verification_stats['files_with_remaining_nan'] += 1
                    
                    print(f"  {var_name}: {nan_count} 個 NaN 值")
                    
                    if var_name == 'sst':
                        verification_stats['total_remaining_sst_nan'] += nan_count
                    elif var_name == 'swh':
                        verification_stats['total_remaining_swh_nan'] += nan_count
            
            verification_stats['files_checked'] += 1
            
        except Exception as e:
            print(f"錯誤檢查 {filename}: {e}")
    
    print("\n" + "=" * 50)
    print("驗證結果統計")
    print("=" * 50)
    print(f"檢查檔案數: {verification_stats['files_checked']}")
    print(f"仍有 NaN 的檔案: {verification_stats['files_with_remaining_nan']}")
    print(f"剩餘 SST NaN: {verification_stats['total_remaining_sst_nan']}")
    print(f"剩餘 SWH NaN: {verification_stats['total_remaining_swh_nan']}")
    
    if verification_stats['files_with_remaining_nan'] == 0:
        print("\n✓ 所有檢查的檔案都沒有 NaN 值！")
    else:
        print(f"\n⚠ 仍有 {verification_stats['files_with_remaining_nan']} 個檔案含有 NaN 值")
    
    print("=" * 50)
    
    return verification_stats

if __name__ == "__main__":
    print("批次處理所有檔案的 NaN 值填補")
    print("=" * 60)
    
    # 處理所有檔案
    print("開始處理所有檔案...")
    processing_stats = process_all_files()
    
    # 驗證結果
    print("\n開始驗證處理結果...")
    verification_stats = verify_random_files(sample_size=50)
    
    print(f"\n🎉 批次處理完成！")
    print(f"✅ 總共處理了 {processing_stats['files_processed']:,} 個檔案")
    print(f"✅ 填補了 {processing_stats['total_sst_filled']:,} 個 SST NaN 值")
    print(f"✅ 填補了 {processing_stats['total_swh_filled']:,} 個 SWH NaN 值")
    if verification_stats['files_with_remaining_nan'] == 0:
        print(f"✅ 驗證通過：隨機檢查的 {verification_stats['files_checked']} 個檔案都沒有 NaN 值")
    else:
        print(f"⚠️  警告：仍有 {verification_stats['files_with_remaining_nan']} 個檔案含有 NaN 值")
