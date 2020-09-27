import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def rawdata_preprocess():
    # loading raw data
    indiv_hit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2016.csv', encoding='cp949')
    indiv_hit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2017.csv', encoding='cp949')
    indiv_hit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2018.csv', encoding='cp949')
    indiv_hit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2019.csv', encoding='cp949')
    indiv_hit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2020.csv', encoding='cp949')
    indiv_pit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2016.csv', encoding='cp949')
    indiv_pit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2017.csv', encoding='cp949')
    indiv_pit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2018.csv', encoding='cp949')
    indiv_pit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2019.csv', encoding='cp949')
    indiv_pit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2020.csv', encoding='cp949')

    team_hit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2016.csv', encoding='cp949')
    team_hit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2017.csv', encoding='cp949')
    team_hit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2018.csv', encoding='cp949')
    team_hit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2019.csv', encoding='cp949')
    team_hit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2020.csv', encoding='cp949')
    team_pit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2016.csv', encoding='cp949')
    team_pit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2017.csv', encoding='cp949')
    team_pit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2018.csv', encoding='cp949')
    team_pit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2019.csv', encoding='cp949')
    team_pit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2020.csv', encoding='cp949')

    # load crawling data
    cr_ind_hit = pd.read_csv('./data/crawling/crawl_ind_h.csv', encoding='cp949')
    cr_ind_pit = pd.read_csv('./data/crawling/crawl_ind_p.csv', encoding='cp949')
    cr_team_hit = pd.read_csv('./data/crawling/crawl_team_h.csv', encoding='cp949')
    cr_team_pit = pd.read_csv('./data/crawling/crawl_team_p.csv', encoding='cp949')

    # concat by theirselves
    indiv_pit = pd.concat([indiv_pit_16, indiv_pit_17, indiv_pit_18, indiv_pit_19, indiv_pit_20])
    indiv_hit = pd.concat([indiv_hit_16, indiv_hit_17, indiv_hit_18, indiv_hit_19, indiv_hit_20])
    team_hit = pd.concat([team_hit_16, team_hit_17, team_hit_18, team_hit_19, team_hit_20])
    team_pit = pd.concat([team_pit_16, team_pit_17, team_pit_18, team_pit_19, team_pit_20])

    # sum BB+IB+HP
    indiv_pit['BB'] = indiv_pit['BB'] + indiv_pit['IB'] + indiv_pit['HP']
    indiv_hit['BB'] = indiv_hit['BB'] + indiv_hit['IB'] + indiv_hit['HP']
    team_pit['BB'] = team_pit['BB'] + team_pit['IB'] + team_pit['HP']
    team_hit['BB'] = team_hit['BB'] + team_hit['IB'] + team_hit['HP']

    # sum HIT+H2+H3
    indiv_pit['HIT'] = indiv_pit['HIT'] + indiv_pit['H2'] + indiv_pit['H3']

    # concat all_data with crawling data
    indiv_pit_all = pd.concat([indiv_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'START_CK',
                                          'RELIEF_CK', 'INN2', 'BF', 'PA', 'AB', 'HIT', 'HR', 'KK', 'BB', 'R',
                                          'ER']],
                               cr_ind_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'START_CK',
                                           'RELIEF_CK', 'INN2', 'BF', 'PA', 'AB', 'HIT', 'HR', 'KK', 'BB', 'R',
                                           'ER']]])

    indiv_hit_all = pd.concat([indiv_hit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'AB', 'RBI',
                                          'RUN', 'HIT', 'H2', 'H3', 'HR', 'BB', 'KK']], cr_ind_hit])

    team_hit_all = pd.concat([team_hit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'AB', 'RBI', 'RUN',
                                        'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK', 'GD', 'ERR', 'LOB']], cr_team_hit])

    team_pit_all = pd.concat([team_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'WLS', 'INN2', 'BF',
                                        'AB', 'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK', 'GD', 'R', 'ER']],
                              cr_team_pit])

    indiv_pit_all = indiv_pit_all[indiv_pit_all['P_ID'] != 'No_Code']
    indiv_pit_all['P_ID'] = indiv_pit_all['P_ID'].astype('int64')
    indiv_hit_all = indiv_hit_all[indiv_hit_all['P_ID'] != 'No_Code']
    indiv_hit_all['P_ID'] = indiv_hit_all['P_ID'].astype('int64')
    return indiv_pit_all, indiv_hit_all, team_hit_all, team_pit_all


def add_and_concat(ind_pit, team_pit, team_hit):
    pitcher_start = ind_pit[ind_pit['START_CK'] == 1].reset_index(drop=True)
    pitcher_start = pitcher_start[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'INN2', 'ER']]
    pitcher_team_all = pd.merge(team_pit, pitcher_start, how='inner',
                                on=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC'],
                                suffixes=('_teampit', '_sp'))

    team_all = pd.merge(pitcher_team_all, team_hit, how='inner',
                        on=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC'],
                        suffixes=('_pit', '_bat'))

    return team_all


def saber_stats(series):
    series['ERA'] = (series['ER_teampit'] / (series['INN2_teampit'] / 3)) * 9

    series['AVG_bat'] = series['HIT_bat'] / series['AB_bat']
    series['AVG_pit'] = series['HIT_pit'] / series['AB_pit']

    series['OPS_bat'] = ((series['HIT_bat'] + series['BB_bat'] + (
                (series['HIT_bat'] - series['H2_bat'] - series['H3_bat'] - series['HR_bat'])
                + (series['H2_bat'] * 2) + (series['H3_bat'] * 3) + (series['HR_bat'] * 4)))) / series['AB_bat']

    series['OPS_pit'] = ((series['HIT_pit'] + series['BB_pit'] + (
                (series['HIT_pit'] - series['H2_pit'] - series['H3_pit'] - series['HR_pit'])
                + (series['H2_pit'] * 2) + (series['H3_pit'] * 3) + (series['HR_pit'] * 4)))) / series['AB_pit']

    series['wOBA_bat'] = ((0.69 * (series['BB_bat']))
                          + (0.89 * (series['HIT_bat'] - series['H2_bat'] - series['H3_bat'] - series['HR_bat']))
                          + (1.27 * series['H2_bat'])
                          + (1.62 * series['H3_bat'])
                          + (2.1 * series['HR_bat'])) / (series['AB_bat'] + series['BB_bat'])

    series['wOBA_pit'] = ((0.69 * (series['BB_pit']))
                          + (0.89 * (series['HIT_pit'] - series['H2_pit'] - series['H3_pit'] - series['HR_pit']))
                          + (1.27 * series['H2_pit'])
                          + (1.62 * series['H3_pit'])
                          + (2.1 * series['HR_pit'])) / (series['AB_pit'] + series['BB_pit'])

    series['FIP'] = (((-2 * series['KK_pit']) + (3 * (series['BB_pit'])) + (13 * series['HR_pit']))
                     / (series['INN2_teampit'] / 3)) + 3.0

    series['WHIP'] = (series['HIT_pit'] + series['BB_pit']) / (series['INN2_teampit'] / 3)

    series['ISO'] = (series['H2_bat'] + (2 * series['H3_bat']) + (3 * series['HR_bat'])) / series['AB_bat']
    series['PE'] = (series['RUN'] ** 2) / ((series['RUN'] ** 2) + (series['R'] ** 2))
    return series

def rel_w_rate(data):
    new = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'wins', 'loses']]
    for k in ['LG', 'OB', 'SS', 'HH', 'HT', 'WO', 'KT', 'SK', 'LT', 'NC']:
        for j in ['LG', 'OB', 'SS', 'HH', 'HT', 'WO', 'KT', 'SK', 'LT', 'NC']:
            if k == j:
                continue
            for i in range(len(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)])):
                if i == 0:
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i,].GDAY_DS, \
                                k, j, 0, 0])

                elif 0 < i < 5:
                    win = list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[:i].WLS.values).count('W')
                    lose = list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[:i].WLS.values).count('L')
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].GDAY_DS, \
                                k, j, win, lose])

                else:
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].GDAY_DS, \
                                k, j, \
                                list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i - 5:i].WLS.values).count(
                                    'W'), \
                                list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i - 5:i].WLS.values).count(
                                    'L')])
    return pd.DataFrame(new[1:], columns=new[0])

def count_of_avg_3(data):
    batter = data[:]
    player_list = list(batter['P_ID'].unique())
    batter = batter[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'P_ID', 'AB', 'HIT']]
    make = pd.DataFrame(columns=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'P_ID', 'AB', 'HIT', 'cum_AB', 'cum_HIT', 'AVG'])
    for i in player_list:
        player = batter[batter['P_ID'] == i]
        player['cum_AB'] = player['AB'].cumsum()
        player['cum_HIT'] = player['HIT'].cumsum()
        player['AVG'] = player['cum_HIT'] / player['cum_AB']
        make = pd.concat([make, player])

    make['AVG'] = make['AVG'].fillna(0)

    team_3hal = pd.DataFrame(columns=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', '3hal'])
    for t in ['NC', 'LT', 'OB', 'SK', 'SS', 'HT', 'HH', 'WO', 'KT', 'LG']:
        temp = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', '3hal']]
        for j in list(make[make['T_ID'] == t].G_ID.unique()):
            ttemp = [j, make[make['G_ID'] == j].GDAY_DS.unique()[0], t,
                     make[(make['G_ID'] == j) & (make['T_ID'] == t)].VS_T_ID.unique()[0]]
            ttemp.append(len(make[(make['G_ID'] == j) & (make['AVG'] >= 0.3) & (make['T_ID'] == t)]))
            temp.append(ttemp)
        temp = pd.DataFrame(temp[1:], columns=temp[0])
        team_3hal = pd.concat([team_3hal, temp])
    return team_3hal

def money():
    #load data
    sal_2016 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2016.csv', encoding='cp949').dropna()
    sal_2017 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2017.csv', encoding='cp949').dropna()
    sal_2018 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2018.csv', encoding='cp949').dropna()
    sal_2019 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2019.csv', encoding='cp949').dropna()
    sal_2020 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2020.csv', encoding='cp949').dropna()


    absal_2016 = sal_2016['MONEY'].str.slice(0,-2).astype(int)
    pcode_2016 = sal_2016[['PCODE','T_ID']]
    result_2016 = pd.concat([pcode_2016,absal_2016], axis=1)

    absal_2017 = sal_2017['MONEY'].str.slice(0,-2).astype(int)
    pcode_2017 = sal_2017[['PCODE','T_ID']]
    result_2017 = pd.concat([pcode_2017,absal_2017], axis=1)

    absal_2018 = sal_2018['MONEY'].str.slice(0,-2).astype(int)
    pcode_2018 = sal_2018[['PCODE','T_ID']]
    result_2018 = pd.concat([pcode_2018,absal_2018], axis=1)

    absal_2019 = sal_2019['MONEY'].str.slice(0,-2).astype(int)
    pcode_2019 = sal_2019[['PCODE','T_ID']]
    result_2019 = pd.concat([pcode_2019,absal_2019], axis=1)

    absal_2020 = sal_2020['MONEY'].str.slice(0,-2).astype(int)
    pcode_2020 = sal_2020[['PCODE','T_ID']]
    result_2020 = pd.concat([pcode_2020,absal_2020], axis=1)

    data_17 = pd.merge(result_2016,result_2017, on ='PCODE', suffixes =('_2016','_2017')).dropna()
    data_17['fluctuation'] = data_17['MONEY_2017']/data_17['MONEY_2016']
    data_18 = pd.merge(result_2017,result_2018, on ='PCODE', suffixes =('_2017','_2018')).dropna()
    data_18['fluctuation'] = data_18['MONEY_2018']/data_18['MONEY_2017']
    data_19 = pd.merge(result_2018,result_2019, on ='PCODE', suffixes =('_2018','_2019')).dropna()
    data_19['fluctuation'] = data_19['MONEY_2019']/data_19['MONEY_2018']
    data_20 = pd.merge(result_2019,result_2020, on ='PCODE', suffixes =('_2019','_2020')).dropna()
    data_20['fluctuation'] = data_20['MONEY_2020']/data_20['MONEY_2019']

    df17 = pd.DataFrame(data_17.groupby('T_ID_2017')['fluctuation'].mean()).reset_index()
    df18 = pd.DataFrame(data_18.groupby('T_ID_2018')['fluctuation'].mean()).reset_index()
    df19 = pd.DataFrame(data_19.groupby('T_ID_2019')['fluctuation'].mean()).reset_index()
    df20 = pd.DataFrame(data_20.groupby('T_ID_2020')['fluctuation'].mean()).reset_index()

    jsn2017 = dict()
    jsn2018 = dict()
    jsn2019 = dict()
    jsn2020 = dict()

    for i in np.array(df17):
        jsn2017[i[0]] = i[1]

    for i in np.array(df18):
        jsn2018[i[0]] = i[1]

    for i in np.array(df19):
        jsn2019[i[0]] = i[1]

    for i in np.array(df20):
        jsn2020[i[0]] = i[1]
    return jsn2017, jsn2018, jsn3019, jsn2020


def main():
    indiv_pit_all, indiv_hit_all, team_hit_all, team_pit_all = rawdata_preprocess()
    indiv_hit_all.to_csv('indiv_hit_all.csv', index=False)
    # team_all = add_and_concat(indiv_pit_all, team_pit_all, team_hit_all)
    # team_saber_all = saber_stats(team_all)
    # team_saber_all.to_csv('./data/result/team_saber.csv', index=False)
    # add_rel = rel_w_rate(team_saber_all)
    # add_rel.to_csv('./data/result/rel_wr.csv', index=False)
    team_3hal = count_of_avg_3(indiv_hit_all)
    team_3hal.to_csv('./data/result/3hal_test.csv', index=False)

if __name__ == '__main__':
    main()