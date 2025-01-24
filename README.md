# Laptop-Price-Prediction

# Separate data for hitflag == 1 and hitflag == 0
hitflag_1_data = train_pred_df[train_pred_df['hitflag'] == 1]
hitflag_0_data = train_pred_df[train_pred_df['hitflag'] == 0]

# Create 10 deciles (adjust the number of bins as needed) for hitflag == 1
hitflag_1_data['pred_tier'] = pd.qcut(hitflag_1_data['train_preds'], 10, labels=False, duplicates='drop')

# Group by pred_tier and calculate aggregates for hitflag == 1
hitflag_1_grouped = hitflag_1_data.groupby('pred_tier').agg({'real': ['sum', 'count']})
hitflag_1_grouped.columns = ['n_pos', 'total']

# Calculate event rates and other metrics for hitflag == 1
hitflag_1_grouped['n_pos_event_rate'] = 100 * hitflag_1_grouped['n_pos'] / hitflag_1_grouped['n_pos'].sum()
hitflag_1_grouped['n_neg'] = hitflag_1_grouped['total'] - hitflag_1_grouped['n_pos']
hitflag_1_grouped['n_neg_event_rate'] = 100 * hitflag_1_grouped['n_neg'] / hitflag_1_grouped['n_neg'].sum()

# Cumulative sums for hitflag == 1
hitflag_1_grouped['n_pos_event_rate_cumm'] = hitflag_1_grouped['n_pos_event_rate'].cumsum()
hitflag_1_grouped['n_neg_event_rate_cumm'] = hitflag_1_grouped['n_neg_event_rate'].cumsum()

# Calculate KS statistic for hitflag == 1
hitflag_1_grouped['KS'] = abs(hitflag_1_grouped['n_neg_event_rate_cumm'] - hitflag_1_grouped['n_pos_event_rate_cumm']) / 100

# Create 10 deciles (adjust the number of bins as needed) for hitflag == 0
hitflag_0_data['pred_tier'] = pd.qcut(hitflag_0_data['train_preds'], 10, labels=False, duplicates='drop')

# Group by pred_tier and calculate aggregates for hitflag == 0
hitflag_0_grouped = hitflag_0_data.groupby('pred_tier').agg({'real': ['sum', 'count']})
hitflag_0_grouped.columns = ['n_pos', 'total']

# Calculate event rates and other metrics for hitflag == 0
hitflag_0_grouped['n_pos_event_rate'] = 100 * hitflag_0_grouped['n_pos'] / hitflag_0_grouped['n_pos'].sum()
hitflag_0_grouped['n_neg'] = hitflag_0_grouped['total'] - hitflag_0_grouped['n_pos']
hitflag_0_grouped['n_neg_event_rate'] = 100 * hitflag_0_grouped['n_neg'] / hitflag_0_grouped['n_neg'].sum()

# Cumulative sums for hitflag == 0
hitflag_0_grouped['n_pos_event_rate_cumm'] = hitflag_0_grouped['n_pos_event_rate'].cumsum()
hitflag_0_grouped['n_neg_event_rate_cumm'] = hitflag_0_grouped['n_neg_event_rate'].cumsum()

# Calculate KS statistic for hitflag == 0
hitflag_0_grouped['KS'] = abs(hitflag_0_grouped['n_neg_event_rate_cumm'] - hitflag_0_grouped['n_pos_event_rate_cumm']) / 100

# Print results for hitflag == 1 and hitflag == 0
print("KS for hitflag == 1:", hitflag_1_grouped['KS'].max())
print(hitflag_1_grouped)
print("KS for hitflag == 0:", hitflag_0_grouped['KS'].max())
print(hitflag_0_grouped)
