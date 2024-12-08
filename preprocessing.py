import numpy as np
import pandas as pd

def preprocessing(data, item_min_count=100, min_len=10, min_duration=30, if_drop_repeats = True,
                  user_id='userid', item_id='itemid', timestamp='timestamp', duration='playevent_play_duration'):
    
    data_ = data.copy()
    data_ = filter_iteractions(data_, min_duration, duration)
    print('filter_iteractions', data_.shape)
    data_ = filter_items(data_, item_min_count, item_id)
    print('filter_items', data_.shape)
    data_ = drop_short_sequences(data_, min_len, user_id)
    print('drop_short_sequences', data_.shape)
    if if_drop_repeats:
        data_ = drop_repeats(data_, user_id, item_id, timestamp)
        print('drop_repeats', data_.shape)
    data_ = drop_short_sequences(data_, min_len, user_id)
    print('drop_short_sequences', data_.shape)
    return data_
    
# def drop_repeats(data, user_id='userid', item_id='itemid', timestamp='timestamp', duration='playevent_play_duration'):
#     data.sort_values(by=[user_id, timestamp], inplace=True)
#     userid = data[user_id].values
#     itemid = data[item_id].values
#     duration = data[duration].values

#     # Создаем маску, чтобы отмечать строки для удаления
#     mask = np.ones(len(data), dtype=bool)

#     # Инициализируем начальные значения
#     max_dur = duration[0]
#     max_ind = 0
#     # Проходим по массиву и отмечаем записи с одинаковыми itemid подряд
#     for i in range(1, len(data)):
#         # Если текущий itemid совпадает с предыдущим для одного и того же пользователя
#         if userid[i] == userid[i - 1]:
#             if itemid[i] == itemid[i - 1]:
#             # Оставляем только запись с максимальным duration
#                 if duration[i] > max_dur:
#                     mask[max_ind] = False
#                     max_ind = i# Убираем предыдущий  максимальный элемент
#                     max_dur = duration[i]
#                 else:
#                     mask[i] = False  # Убираем текущий элемент
#             else:
#                 max_dur = duration[i]
#                 max_ind = i  
#         else:
#             max_dur = duration[i]
#             max_ind = i
#     # Применяем маску к DataFrame
#     return data[mask]

def drop_repeats(data, user_id='user_id', item_id="item_id", timestamp="timestamp"):
    """Remove repeated items like i-i-j -> i-j."""

    data.sort_values([user_id, timestamp], inplace=True)
    data['user_item'] = data[user_id].astype(str) + '_' + data[item_id].astype(str)

    while (data['user_item'].shift() == data['user_item']).sum() != 0:
        not_duplicates_ind = data['user_item'].shift() != data['user_item']
        data = data.loc[not_duplicates_ind]

    data = data.drop('user_item', axis=1)

    return data

def filter_items(data, item_min_count, item_id="itemid"): 
    item_count = data.groupby(by=[item_id])[item_id].count()
    used_items = item_count[item_count.values>item_min_count].index.values 
    data_ = data.loc[data[item_id].isin(used_items)]
    return data_
    
    
def drop_short_sequences(data, min_len, user_id='userid'):    
    user_count = data.groupby(by=[user_id])[user_id].count()
    used_users = user_count[user_count.values>min_len].index.values
    data_ = data.loc[data[user_id].isin(used_users)]
    return data_

def filter_iteractions(data, min_duration, duration='playevent_play_duration'):
    data_ = data[data[duration] > min_duration]
    return data_