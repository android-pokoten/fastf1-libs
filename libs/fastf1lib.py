class myFastf1:

    # キャッシュのパスを定義。呼び出し先のパスが変わることを考慮して絶対パスで記述する。
    # セッション用の変数は未使用
    def __init__(self):
        self.session_race = ''
        self.session_qual = ''
        self.cache = '/work/fastf1/cache/'

    # 【旧版】セッション読み込み
    def load_session(self, name, year):
        import fastf1

        fastf1.Cache.enable_cache(self.cache)
        self.session_race = fastf1.get_session(year, name, 'R')
        self.session_race.load()
        self.session_qual = fastf1.get_session(year, name, 'Q')
        self.session_qual.load()

    def load_session_o(self, name, year, s):
        import fastf1

        fastf1.Cache.enable_cache(self.cache)
        session = fastf1.get_session(year, name, s)
        session.load()

        return session

    def speed_traces(self, session):
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


    def tyre_strategies(self, session):
        import fastf1.plotting
        from matplotlib import pyplot as plt

        laps = session.laps

        drivers = session.drivers
        drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

        stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
        stints = stints.groupby(["Driver", "Stint", "Compound"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        stints['Compound'].loc[stints['Compound'] == 'TEST_UNKNOWN'] = 'TEST-UNKNOWN'

        fig, ax = plt.subplots(figsize=(5, 10))

        for driver in drivers:
            driver_stints = stints.loc[stints["Driver"] == driver]

            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                plt.barh(
                    y=driver,
                    width=row["StintLength"],
                    left=previous_stint_end,
                    color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                    edgecolor="black",
                    fill=True
                )

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

    def driver_laptime(self, session, driver):
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        driver_laps = session.laps.pick_driver(driver).pick_quicklaps().reset_index()
        driver_laps['Compound'].loc[driver_laps['Compound'] == 'TEST_UNKNOWN'] = 'TEST-UNKNOWN'

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

        plt.tight_layout()
        plt.show()

    def laptime_distribution(self, session):
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        point_finishers = session.drivers[:10]
        driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps()
        driver_laps = driver_laps.reset_index()
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

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

    def speed_compare(self, session, d1, d2):
        import fastf1.plotting
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        d1_lap = session.laps.pick_driver(d1).pick_fastest()
        d2_lap = session.laps.pick_driver(d2).pick_fastest()

        d1_tel = d1_lap.get_car_data().add_distance()
        d2_tel = d2_lap.get_car_data().add_distance()
        
        d1_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[d1]]
        d2_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[d2]]

        fig, ax = plt.subplots()
        ax.plot(d1_tel['Distance'], d1_tel['Speed'], color=d1_color, label=d1)
        ax.plot(d2_tel['Distance'], d2_tel['Speed'], color=d2_color, label=d2)
        
        ax.set_xlabel('Distance in m')
        ax.set_ylabel('Speed in km/h')

        ax.legend()
        plt.suptitle(f"Fastest Lap Comparison \n "
                     f"{session.event['EventName']} {session.event.year} {session.name}")
        plt.show()        

    def team_comparison(self, session):
        import fastf1
        import fastf1.plotting
        import seaborn as sns
        import matplotlib.pyplot as plt

        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

        laps = session.laps.pick_quicklaps()

        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

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


    def minisector_compare(self, session, d1, d2):
        from fastf1 import plotting
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        from matplotlib.collections import LineCollection
        from matplotlib import cm
        import numpy as np
        import pandas as pd

        plotting.setup_mpl()
        pd.options.mode.chained_assignment = None
        
        alaps = session.laps.pick_quicklaps()
        drivers = [d1, d2]

        d1_lap = session.laps.pick_driver(d1).pick_fastest()
        d2_lap = session.laps.pick_driver(d2).pick_fastest()

        d1_tel = d1_lap.get_telemetry().add_distance()
        d1_tel['Driver'] = d1
        d2_tel = d2_lap.get_telemetry().add_distance()
        d2_tel['Driver'] = d2

        d1_color = plotting.DRIVER_COLORS[plotting.DRIVER_TRANSLATE[d1]]
        d2_color = plotting.DRIVER_COLORS[plotting.DRIVER_TRANSLATE[d2]]

        telemetry = pd.DataFrame()
        telemetry = pd.concat([telemetry, d1_tel], ignore_index=True, axis=0)
        telemetry = pd.concat([telemetry, d2_tel], ignore_index=True, axis=0)

        telemetry = telemetry[['Distance', 'Driver', 'Speed', 'X', 'Y']]

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

        average_speed = telemetry.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()

        fastest_compound = average_speed.loc[average_speed.groupby(['Minisector'])['Speed'].idxmax()]

        fastest_compound = fastest_compound[['Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_drv'})

        telemetry = telemetry.merge(fastest_compound, on=['Minisector'])

        telemetry = telemetry.sort_values(by=['Distance'])

        telemetry.loc[telemetry['Fastest_drv'] == d1, 'Fastest_drv_int'] = d1_color
        telemetry.loc[telemetry['Fastest_drv'] == d2, 'Fastest_drv_int'] = d2_color

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

        plt.suptitle(f"Minisector {d1} vs {d2}")
        plt.gca().add_collection(lc_comp)
        plt.axis('equal')
        plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        plt.show()


    def slick_vs_wet(self, session, target_lap):
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
        import fastf1.plotting
        import matplotlib.pyplot as plt
        import numpy as np

        fastf1.plotting.setup_mpl(misc_mpl_mods=False)

        fig, ax = plt.subplots()

        for drv in drivers:
            drv_laps = session.laps.pick_driver(drv)
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
        plt.suptitle(f"{session.event['EventName']} {session.event.year} Race Laptime")

        plt.tight_layout()
        plt.show()

    def deltatime_comperition(self, session, drivers):
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
