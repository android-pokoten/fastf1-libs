class myFastf1:
    # target fastf1 ver: 3.1.3

    # キャッシュのパスを定義。呼び出し先のパスが変わることを考慮して絶対パスで記述する。
    # セッション用の変数は未使用
    def __init__(self):
        self.session_race = ''
        self.session_qual = ''

    # 【旧版】セッション読み込み
    def load_session(self, name, year, cache = './cache'):
        import fastf1

        fastf1.Cache.enable_cache(cache)
        self.session_race = fastf1.get_session(year, name, 'R')
        self.session_race.load()
        self.session_qual = fastf1.get_session(year, name, 'Q')
        self.session_qual.load()

    def load_session_o(self, name, year, s, cache = './cache'):
        """概要
        指定したセッションを読み込み、セッションオブジェクトを返す。

        Parameters
        ----------
        name: string
            イベント名
        year: int
            開催年
        s : string
            セッション(後述)
        cache: string
            キャッシュパス

        セッションは以下の文字列を使用するか、数字 (FP1 が 1、FP2 が 2、レースは 5 など) で指定する。
        Race: 'R'
        Qualify: 'Q'
        Sprint： 'S'
        Sprint Shootout： 'SQ'
        FP1～3: 'FP1', 'FP2', 'FP3'

        Returns
        -------
        session: 
            セッションオブジェクト
        """
        import fastf1

        fastf1.Cache.enable_cache(cache)
        session = fastf1.get_session(year, name, s)
        session.load()

        return session

    def speed_traces(self, session):
        """
        1 ラップの速度をコーナー位置ありでグラフ化する。
        #### Parameters ####
        session : session
            セッションオブジェクト
        """
        import fastf1.plotting
        from matplotlib import pyplot as plt

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        fastest_lap = session.laps.pick_fastest()
        car_data = fastest_lap.get_car_data().add_distance()

        circuit_info = session.get_circuit_info()

        team_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[fastest_lap['Driver']]]

        fig, ax = plt.subplots()
        ax.plot(car_data['Distance'], car_data['Speed'], color=team_color, label=fastest_lap['Driver'])

        v_min = car_data['Speed'].min()
        v_max = car_data['Speed'].max()
        ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20, linestyles='dotted', color='gray')

        for _, corner in circuit_info.corners.iterrows():
            txt = f"{corner['Number']}{corner['Letter']}"
            ax.text(corner['Distance'], v_min-30, txt, va='center_baseline', ha='center', size='small')

        ax.set_xlabel('Distance in m')
        ax.set_ylabel('Speed in lm/h')
        ax.legend()

        ax.set_ylim([v_min - 40, v_max + 20])

        plt.show()

    def position_changes(self, session):
        """
        ## セッション中のポジション推移 ##

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        """
        import fastf1.plotting
        from matplotlib import pyplot as plt

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        fig, ax = plt.subplots(figsize=(8.0, 4.9))

        for drv in session.drivers:
            drv_laps = session.laps.pick_driver(drv)
            if len(drv_laps) == 0:
                continue
            abb = drv_laps['Driver'].iloc[0]
            color = fastf1.plotting.driver_color(abb)
            ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, color=color)
        
        ax.set_ylim([20.5, 0.5])
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_xlabel('Lap')
        ax.set_ylabel('Position')

        ax.legend(bbox_to_anchor=(1.0, 1.02))
        plt.tight_layout()
        plt.show()


    def tyre_strategies(self, session, drivers):
        """
        ## タイヤ使用履歴 ##

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        2. drivers : list
            一覧表示するドライバーのリスト。全員を指定する場合は session_race.drivers を渡す
        """
        import fastf1.plotting
        from matplotlib import pyplot as plt

        laps = session.laps

        drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

        stints = laps[["Driver", "Stint", "Compound", "LapNumber", "FreshTyre"]]
        stints = stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})

        hx = len(drivers) / 2
        fig, ax = plt.subplots(figsize=(5, hx))

        for driver in drivers:
            driver_stints = stints.loc[stints["Driver"] == driver]

            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                colors = 'black' if row["FreshTyre"] else 'blue'
                p = plt.barh(
                    y=driver,
                    width=row["StintLength"],
                    left=previous_stint_end,
                    color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                    edgecolor='black',
                    fill=True
                )
                ax.bar_label(p, label_type='center', color=colors)

                previous_stint_end += row["StintLength"]

        plt.suptitle(f"{session.event['EventName']} {session.event.year} - {session.name} Tyre Strategies")
        plt.xlabel("Lap Number")
        plt.grid(False)

        ax.invert_yaxis()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def quali_result(self, session):
        """
        ## 予選のパフォーマンス差 ##

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from timple.timedelta import strftimedelta
        import numpy as np

        import fastf1
        import fastf1.plotting
        from fastf1.core import Laps

        fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=False)

        drivers = pd.unique(session.laps['Driver'])

        list_fastest_laps = list()
        for drv in drivers:
            drv_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
            if drv_fastest_lap['Driver'] is np.nan:
                continue
            list_fastest_laps.append(drv_fastest_lap)
        fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

        pole_lap = fastest_laps.pick_fastest()
        fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

        team_colors = list()
        for index, lap in fastest_laps.iterlaps():
            color = fastf1.plotting.team_color(lap['Team'])
            team_colors.append(color)

        fig, ax = plt.subplots()
        ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'], color=team_colors, edgecolor='grey')
        ax.set_yticks(fastest_laps.index)
        ax.set_yticklabels(fastest_laps['Driver'])

        ax.invert_yaxis()

        ax.set_axisbelow(True)
        ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

        lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

        plt.suptitle(f"{session.event['EventName']} {session.event.year} {session.name}\n"
                     f"Fastest Lap: {lap_time_string} {(pole_lap['Driver'])}")
        
        plt.show()

    def driver_laptime(self, session, driver, min_sec=0, max_sec=0):
        """
        ## ドライバーのラップタイム一覧 ##
        タイヤコンパウントで色分けしてラップタイムをプロットする。  

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. driver : string
            ドライバー名を3文字略称で指定
        1. min_sec : int
            グラフの最小値(秒数)
        1. max_sec : int
            グラフの最大値(秒数)
        """
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        driver_laps = session.laps.pick_driver(driver).pick_quicklaps().reset_index(drop=True)
        #driver_laps['Compound'].loc[driver_laps['Compound'] == 'TEST_UNKNOWN'] = 'TEST-UNKNOWN'

        fig, ax = plt.subplots(figsize=(8, 8))

        sns.scatterplot(data=driver_laps,
                        x="LapNumber",
                        y="LapTime",
                        ax=ax,
                        hue="Compound",
                        palette=fastf1.plotting.COMPOUND_COLORS,
                        s=80,
                        linewidth=0,
                        legend='auto')
        
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time")

        ax.invert_yaxis()
        plt.suptitle(f"{session.event['EventName']} {session.event.year} {driver} {session.name} Laptime")

        plt.grid(color='w', which='major', axis='both')
        sns.despine(left=True, bottom=True)

        if not min_sec == 0:
            ax.set_ylim(np.timedelta64(min_sec, 's'), np.timedelta64(max_sec, 's'))

        plt.tight_layout()
        plt.show()

    def laptime_distribution(self, session):
        """
        ## 各ドライバーのラップタイムの出現範囲をプロット ##
        上位10名について表示する。  

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        """
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        point_finishers = session.drivers[:10]
        driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps()
        driver_laps = driver_laps.reset_index()
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
        driver_laps = driver_laps.reset_index(drop=True)

        finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]

        driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(data=driver_laps,
                       x="Driver",
                       y="LapTime(s)",
                       inner=None,
                       scale="area",
                       order=finishing_order,
                       palette=driver_colors,
                       ax=ax
                       )
        
        sns.swarmplot(data=driver_laps,
                      x="Driver",
                      y="LapTime(s)",
                      order=finishing_order,
                      hue="Compound",
                      palette=fastf1.plotting.COMPOUND_COLORS,
                      hue_order=["SOFT", "MEDIUM", "HARD"],
                      linewidth=0,
                      size=5,
                      ax=ax
                      )
        
        ax.set_xlabel("Driver")
        ax.set_ylabel("Lap Time (s)")
        plt.suptitle("Lap Time Distributions")
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        plt.show()

    def speed_compare(self, session, driver1, driver2):
        import fastf1.plotting
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        d1_lap = session.laps.pick_driver(driver1).pick_fastest()
        d2_lap = session.laps.pick_driver(driver2).pick_fastest()

        d1_tel = d1_lap.get_car_data().add_distance()
        d2_tel = d2_lap.get_car_data().add_distance()
        
        d1_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[driver1]]
        d2_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[driver2]]

        fig, ax = plt.subplots()
        ax.plot(d1_tel['Distance'], d1_tel['Speed'], color=d1_color, label=driver1)
        ax.plot(d2_tel['Distance'], d2_tel['Speed'], color=d2_color, label=driver2)
        
        ax.set_xlabel('Distance in m')
        ax.set_ylabel('Speed in km/h')

        ax.legend()
        plt.suptitle(f"Fastest Lap Comparison \n "
                     f"{session.event['EventName']} {session.event.year} {session.name}")
        plt.show()        

    def team_comparison(self, session):
        """
        ## チームごとのパフォーマンス差 ##

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        """
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        laps = session.laps.pick_quicklaps()

        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        transformed_laps = transformed_laps.reset_index(drop=True)

        team_order = (
            transformed_laps[["Team", "LapTime (s)"]]
            .groupby("Team")
            .median()["LapTime (s)"]
            .sort_values()
            .index
        )

        team_palette = {team: fastf1.plotting.team_color(team) for team in team_order}

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(
            data=transformed_laps,
            x="Team",
            y="LapTime (s)",
            order=team_order,
            palette=team_palette,
            whiskerprops=dict(color="white"),
            boxprops=dict(edgecolor="white"),
            medianprops=dict(color="grey"),
            capprops=dict(color="white"),
        )

        plt.suptitle(f"Team Pace Comparison \n "
                     f"{session.event['EventName']} {session.event.year} {session.name}")
        plt.grid(visible=False)

        ax.set(xlabel=None)
        plt.tight_layout()
        plt.show()


    def minisector_compare(self, session, driver1, driver2, minisectors):
        """
        ## ミニセクターごとにドライバー比較 ##
        ミニセクター区切りは、最初に 0 を入れること。  
        最後はラップ終わりにする必要はない。最後に指定した値から、ラップ最後までが最後のミニセクターになる。  
        等間隔で区切った値を使う場合は、以下のコードを参考に計算して引数で渡すことが可能。

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. driver1 : string
            ドライバー1。
        1. driver2 : string
            ドライバー2。
        1. minisectors : [int]
            ミニセクターの距離をリストで指定。
        """
        from fastf1 import plotting
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        from matplotlib.collections import LineCollection
        from matplotlib import cm
        import numpy as np
        import pandas as pd

        plotting.setup_mpl()
        pd.options.mode.chained_assignment = None

        d1_lap = session.laps.pick_driver(driver1).pick_fastest()
        d2_lap = session.laps.pick_driver(driver2).pick_fastest()

        d1_tel = d1_lap.get_telemetry().add_distance()
        d1_tel['Driver'] = driver1
        d2_tel = d2_lap.get_telemetry().add_distance()
        d2_tel['Driver'] = driver2

        d1_color = plotting.DRIVER_COLORS[plotting.DRIVER_TRANSLATE[driver1]]
        d2_color = plotting.DRIVER_COLORS[plotting.DRIVER_TRANSLATE[driver2]]

        telemetry = pd.DataFrame()
        telemetry = pd.concat([telemetry, d1_tel], ignore_index=True, axis=0)
        telemetry = pd.concat([telemetry, d2_tel], ignore_index=True, axis=0)

        telemetry = telemetry[['Distance', 'Driver', 'Speed', 'X', 'Y']]

        telemetry['Minisector'] = telemetry['Distance'].apply(
            lambda z: (
                minisectors.index(
                    min(minisectors, key=lambda x: abs(x-z))
                )+1
            )
        )

        average_speed = telemetry.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()

        fastest_compound = average_speed.loc[average_speed.groupby(['Minisector'])['Speed'].idxmax()]

        fastest_compound = fastest_compound[['Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_drv'})

        telemetry = telemetry.merge(fastest_compound, on=['Minisector'])

        telemetry = telemetry.sort_values(by=['Distance'])

        telemetry.loc[telemetry['Fastest_drv'] == driver1, 'Fastest_drv_int'] = d1_color
        telemetry.loc[telemetry['Fastest_drv'] == driver2, 'Fastest_drv_int'] = d2_color

        single_lap = telemetry

        x = np.array(single_lap['X'].values)
        y = np.array(single_lap['Y'].values)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        compound = single_lap['Fastest_drv_int'].to_numpy()

        cmap = cm.get_cmap('ocean', 2)
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), colors=compound)
        lc_comp.set_linewidth(2)

        plt.rcParams['figure.figsize'] = [12, 5]

        plt.suptitle(f"Minisector {driver1} vs {driver2}")
        plt.gca().add_collection(lc_comp)
        plt.axis('equal')
        plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        plt.show()


    def slick_vs_wet(self, session, target_lap):
        """
        ## スリック vs レインの色付け ##
        セッション中に雨が降った時に、ミニセクターごとにスリックタイヤとレインタイヤで早いほうを色付けする。

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. target_lap : int
            比較する周回
        """
        from fastf1 import plotting
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        from matplotlib.collections import LineCollection
        from matplotlib import cm
        import numpy as np
        import pandas as pd

        plotting.setup_mpl()
        pd.options.mode.chained_assignment = None
        
        alap = session.laps.pick_lap(target_lap)
        alaps = session.laps.pick_quicklaps()
        drivers = pd.unique(alap['Driver'])

        telemetry = pd.DataFrame()

        for driver in drivers:
            driver_laps = alap.pick_driver(driver)

            for lap in driver_laps.iterlaps():
                driver_telemetry = lap[1].get_telemetry().add_distance()
                driver_telemetry['Driver'] = driver
                driver_telemetry['Lap'] = lap[1]['LapNumber']
                driver_telemetry['Compound'] = lap[1]['Compound']

                telemetry = pd.concat([telemetry, driver_telemetry], ignore_index=True, axis=0)

        telemetry = telemetry[['Lap', 'Distance', 'Compound', 'Speed', 'X', 'Y']]
        telemetry['Compound'].loc[telemetry['Compound'] != 'INTERMEDIATE'] = 'SLICK'

        num_minisectors = 25

        total_distance = max(telemetry['Distance'])

        minisector_length = total_distance / num_minisectors

        minisectors = [0]

        for i in range(0, (num_minisectors - 1)):
            minisectors.append(minisector_length * (i + 1))

        telemetry['Minisector'] = telemetry['Distance'].apply(
            lambda z: (
                minisectors.index(
                    min(minisectors, key=lambda x: abs(x-z))
                )+1
            )
        )

        average_speed = telemetry.groupby(['Lap', 'Minisector', 'Compound'])['Speed'].mean().reset_index()

        fastest_compound = average_speed.loc[average_speed.groupby(['Minisector'])['Speed'].idxmax()]

        fastest_compound = fastest_compound[['Minisector', 'Compound']].rename(columns={'Compound': 'Fastest_compound'})

        telemetry = telemetry.merge(fastest_compound, on=['Minisector'])

        telemetry = telemetry.sort_values(by=['Distance'])

        telemetry.loc[telemetry['Fastest_compound'] == 'INTERMEDIATE', 'Fastest_compound_int'] = 1
        telemetry.loc[telemetry['Fastest_compound'] == 'SLICK', 'Fastest_compound_int'] = 2
        #single_lap = telemetry.loc[telemetry['Lap'] == 27]

        x = np.array(telemetry['X'].values)
        y = np.array(telemetry['Y'].values)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        compound = telemetry['Fastest_compound_int'].to_numpy()

        cmap = cm.get_cmap('ocean', 2)
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
        lc_comp.set_array(compound)
        lc_comp.set_linewidth(2)

        plt.rcParams['figure.figsize'] = [12, 5]

        plt.gca().add_collection(lc_comp)
        plt.axis('equal')
        plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        plt.show()

    def laptime_comperition(self, session, drivers, min_sec, max_sec):
        """
        ## 各ドライバーのラップタイム推移 ##
        ドライバーリストは [] で囲んで , で区切って記載する。  

        #### Parameters ####
        ----------
        1. session : session
            セッションオブジェクト
        1. drivers : [string]
            プロットするドライバーをリスト形式で指定
        1. min_sec : int
            y 軸の上端値(秒数で指定)
        1. max_sec : int
            y 軸の下端値(秒数で指定)
        """
        self.laptime_comperition_sep(session.event, session.laps, drivers, min_sec, max_sec)

    def laptime_comperition_sep(self, event, laps, drivers, min_sec, max_sec):
        """
        ## 各ドライバーのラップタイム推移(一部のラップのみ) ##
        ソフトタイヤのラップのみをグラフ化する場合など、別途フィルターしたラップでグラフ化したい場合に使用する。
        ドライバーリストは [] で囲んで , で区切って記載する。  

        #### Parameters ####
        ----------
        1. event : session.event 
            セッションのeventオブジェクト
        1. laps : session.laps 
            セッションのlapsオブジェクト。セッション全体をグラフ化する場合は session.laps で指定する。
            一部のラップのみをグラフ化する場合は、先に条件でラップをフィルターしてから指定する。
        1. drivers : [string]
            プロットするドライバーをリスト形式で指定
        1. min_sec : int
            y 軸の上端値(秒数で指定)
        1. max_sec : int
            y 軸の下端値(秒数で指定)
        """
        import fastf1.plotting
        import matplotlib.pyplot as plt
        import numpy as np

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        fig, ax = plt.subplots()

        for drv in drivers:
            drv_laps = laps.pick_driver(drv)
            if len(drv_laps) == 0:
                continue
            abb = drv_laps['Driver'].iloc[0]
            color = fastf1.plotting.driver_color(abb)
            ax.plot(drv_laps['LapNumber'], drv_laps['LapTime'], label=abb, color=color)


        ax.legend()
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time")
        ax.yaxis.grid(True, which='major', linestyle='--', color='gray', zorder=-1000)
        ax.set_ylim(np.timedelta64(min_sec, 's'), np.timedelta64(max_sec, 's'))

        ax.invert_yaxis()
        plt.suptitle(f"{event['EventName']} {event.year} Laptime Comperition")

        plt.tight_layout()
        plt.show()

    def deltatime_comperition(self, session, drivers):
        """
        ## 複数ドライバーのラップタイム推移を比較 ##
        ドライバーリストは、[] で囲んで , で区切って記載する。  
        ドライバーリストの最初に指定したドライバーを基準タイム(0の線)として表示する。

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. drivers : [string]
            ドライバーのリスト
        """
        import fastf1.plotting
        import matplotlib.pyplot as plt
        import numpy as np

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        target = session.laps.pick_driver(drivers[0])['LapStartTime'].dt.total_seconds().reset_index(drop=True)
        target_drv = session.laps.pick_driver(drivers[0])['Driver'].iloc[0]

        fig, ax = plt.subplots()
        for drv in drivers:
            drv_laps = session.laps.pick_driver(drv)['LapStartTime'].dt.total_seconds().reset_index(drop=True)
            if len(drv_laps) == 0:
                continue
            abb = session.laps.pick_driver(drv)['Driver'].iloc[0]
            color = fastf1.plotting.driver_color(abb)
            ax.plot(target - drv_laps, label=abb, color=color)


        ax.legend()
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Time Delta")
        ax.yaxis.grid(True, which='major', linestyle='--', color='gray', zorder=-1000)
        #ax.set_ylim(np.timedelta64(min_sec, 's'), np.timedelta64(max_sec, 's'))

        plt.suptitle(f"{session.event['EventName']} {session.event.year} Time Delta from {target_drv}")

        plt.tight_layout()
        plt.show()

    def speedtrap_heatmap(self, session, datanum, vmin):
        """
        ## 最高速度のヒートマップ ##

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. datanum : int
            データ個数
        1. vmin : int
            最小速度
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        drivers = session.drivers

        df = pd.DataFrame()

        for drv in drivers:
            laps = session.laps.pick_driver(drv)

            speed_list = list()
            for index, lap in laps.iterlaps():
                speed = max(lap.get_car_data()['Speed'])
                speed_list.append(speed)
            speed_list.sort()
            speed_list.reverse()
            speed_list = speed_list[:datanum]

            abb = session.get_driver(drv)['Abbreviation']

            df1 = pd.DataFrame({abb: speed_list})

            df = pd.concat([df, df1.T])

        df.fillna(0, inplace=True)
        df = df.astype('int64')

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            df,
            annot=True,
            fmt="d",
            #annot_kws={"fontsize": 16, "fontfamily": "serif"},
            #square=True,
            cmap='hot',
            vmin=vmin,
            ax=ax
            )
        plt.suptitle(f"{session.event['EventName']} {session.event.year} Speed Trap Heatmap")
        plt.show()

    def datachart_compare(self, session, driver1, driver2):
        """
        2ドライバーのテレメトリーデータ比較。
        ドライバーの指定は、3文字省略形、またはカーナンバー

        Parameters
        ----------
        session : session
            セッションオブジェクト
        driver1 : string
            ドライバー1
        driver2 : string
            ドライバー2
        """
        import fastf1
        from fastf1.core import Laps
        import fastf1.plotting

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib import cm
        import numpy as np
        import pandas as pd
        from timple.timedelta import strftimedelta

        abb1 = session.laps.pick_driver(driver1)['Driver'].iloc[0]
        color1 = fastf1.plotting.driver_color(abb1)

        abb2 = session.laps.pick_driver(driver2)['Driver'].iloc[0]
        color2 = fastf1.plotting.driver_color(abb2)

        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        tel1 = lap1.get_telemetry()
        linestyle1 = 'solid'

        lap2 = session.laps.pick_driver(driver2).pick_fastest()
        tel2 = lap2.get_telemetry()
        linestyle2 = 'dashed'

        fig = plt.figure(figsize=[10, 12])
        axes = fig.subplots(6, 1)

        axes[0].plot(tel1["Distance"], tel1["Speed"], color=color1, linestyle=linestyle1)
        axes[0].plot(tel2["Distance"], tel2["Speed"], color=color2, linestyle=linestyle2)
        axes[0].grid(which = 'major', color='gray', linestyle='--' )
        axes[0].set_ylabel('Speed (km/h)')

        axes[1].plot(tel1["Distance"], tel1["Throttle"], color=color1, linestyle=linestyle1)
        axes[1].plot(tel2["Distance"], tel2["Throttle"], color=color2, linestyle=linestyle2)
        axes[1].grid(which = 'major', color='gray', linestyle='--' )
        axes[1].set_ylabel('Throttle (%)')

        axes[2].plot(tel1["Distance"], tel1["Brake"], color=color1, linestyle=linestyle1)
        axes[2].plot(tel2["Distance"], tel2["Brake"], color=color2, linestyle=linestyle2)
        axes[2].grid(which = 'major', color='gray', linestyle='--' )
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["OFF", "ON"])
        axes[2].set_ylabel('Brake')

        axes[3].plot(tel1["Distance"], tel1["RPM"], color=color1, linestyle=linestyle1)
        axes[3].plot(tel2["Distance"], tel2["RPM"], color=color2, linestyle=linestyle2)
        axes[3].grid(which = 'major', color='gray', linestyle='--' )
        axes[5].set_yticks([9000, 12000])
        axes[3].set_ylabel('RPM')

        axes[4].plot(tel1["Distance"], tel1["nGear"], color=color1, linestyle=linestyle1)
        axes[4].plot(tel2["Distance"], tel2["nGear"], color=color2, linestyle=linestyle2)
        axes[4].grid(which = 'major', color='gray', linestyle='--' )
        axes[4].set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
        axes[4].set_ylabel('Gear')

        axes[5].plot(tel1["Distance"], tel1["DRS"], color=color1, linestyle=linestyle1)
        axes[5].plot(tel2["Distance"], tel2["DRS"], color=color2, linestyle=linestyle2)
        axes[5].set_xlabel('Distance (m)')
        axes[5].grid(which = 'major', color='gray', linestyle='--' )
        axes[5].set_yticks([8, 12])
        axes[5].set_yticklabels(["OFF", "ON"])
        axes[5].set_ylabel('DRS')

        lap_time_string = strftimedelta(lap1['LapTime'], '%m:%s.%ms')
        lap_time2_string = strftimedelta(lap2['LapTime'], '%m:%s.%ms')
        plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                    f"Fastest Lap: {lap_time_string} ({lap1['Driver']}) vs {lap_time2_string} ({lap2['Driver']})")
        plt.show()


    def gap_to_average(self, session, drivers, min_sec, max_sec, lap_num=0):
        """
        ## 平均タイムとのタイム差 ##
        ドライバーリストは [] で囲んで , で区切って記載する。  
        ドライバーリストの先頭の平均ラップタイムを 0 とする。  
        セーフティーカー先導等、遅いラップを挟むと平均値が上手く計算できない。赤旗中断を挟むと意図したグラフにならない。

        #### Parameters ####
        ----------
        1. session : session 
            セッションオブジェクト
        1. drivers : [string]
            プロットするドライバーをリスト形式で指定
        1. min_sec : int
            y 軸の上端値
        1. max_sec : int
            y 軸の下端値
        1. lap_num : int
            描写を開始する周回数。初期値:0
        """
        import fastf1.plotting
        import matplotlib.pyplot as plt
        import numpy as np

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        fig, ax = plt.subplots()

        target = session.laps.pick_driver(drivers[0])[lap_num:]
        target = target.reset_index()
        target['LapTimeSec'] = target['LapTime'].dt.total_seconds()
        #lap_count = target['LapNumber'].count()
        lap_count = target['LapTime'].count()
        target_avg = target['LapTimeSec'].sum() / lap_count

        for drv in drivers:
            drv_laps = session.laps.pick_driver(drv)[lap_num:]
            drv_laps = drv_laps.reset_index()
            if len(drv_laps) == 0:
                continue
            abb = drv_laps['Driver'].iloc[0]
            color = fastf1.plotting.driver_color(abb)
            drv_laps['LapTimeSec'] = drv_laps['LapTime'].dt.total_seconds()
            drv_laps['deltaLapTime'] = target_avg - drv_laps['LapTimeSec']
            drv_laps.loc[0, 'deltaLapTime'] = drv_laps.loc[0, 'deltaLapTime'] - (drv_laps.loc[0, 'LapStartTime'].total_seconds() - target.loc[0, 'LapStartTime'].total_seconds())
            drv_laps['deltaCumsum'] = drv_laps['deltaLapTime'].cumsum()

            ax.plot(drv_laps['LapNumber'], drv_laps['deltaCumsum'], label=abb, color=color)


        ax.legend()
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time")
        ax.yaxis.grid(True, which='major', linestyle='--', color='gray', zorder=-1000)
        ax.set_ylim([min_sec, max_sec])

        ax.invert_yaxis()
        plt.suptitle(f"{session.event['EventName']} {session.event.year} Gap to Average")

        plt.tight_layout()
        plt.show()


    def cornerspeed_compare(self, session, driver1, driver2, min_dist, max_dist):
        """
        2 ドライバー間のコーナーの速度差を計算する。

        コーナー位置は、以下で確認可能
        circuit_info = session_fp2.get_circuit_info()
        circuit_info.corners
        -> 出力の Distance が距離

        Parameters
        ----------
        session : session
            セッションオブジェクト
        driver1 : string
            ドライバー名(3文字略称)
        driver2 : string
            ドライバー名(3文字略称)
        min_dist : int
            表示開始の距離
        max_dist : int
            表示終了の距離
        """
        import fastf1.plotting
        import matplotlib.pyplot as plt

        # ブレーキ、フルスロットルを表示するために twinx() した ax1 を用意
        fig, ax = plt.subplots()
        ax1 = ax.twinx()

        # ドライバーごとにグラフ位置をずらすためのシフト値
        index = 0
        for drv in [driver1, driver2]:
            lap1 = session.laps.pick_driver(drv).pick_fastest()
            color1 = fastf1.plotting.driver_color(lap1['Driver'])
            tel1 = lap1.get_telemetry()
            # 指定した距離を抽出するための条件
            condition = (tel1['Distance'] >= min_dist) & (tel1['Distance'] <= max_dist)

            # 速度グラフを表示
            ax.plot(tel1["Distance"].loc[condition], tel1["Speed"].loc[condition], color=color1)

            # スロットル開度 90 以上 (全開が 100 ではなく 99 のデータもあったため)、またはブレーキを踏んでいる時を 1 にする Inaction を計算する
            tel1['Inaction'] = ((tel1['Throttle'] > 90) | (tel1['Brake'])).astype(int) + index
            ax1.plot(tel1["Distance"].loc[condition], tel1["Inaction"].loc[condition], color=color1, linestyle='dashed')

            index = index + 1
            
        ax.grid(which = 'major', color='gray', linestyle='--' )
        ax.set_ylabel('Speed (km/h)')
        ax1.set_ylim([-5, 5])

        ax.legend()
        plt.suptitle(f"{session.event['EventName']} {session.event.year}\n"
                     f"{driver1} vs {driver2}")
        plt.show()
