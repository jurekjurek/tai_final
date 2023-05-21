for frame in df_input_20[0:10]:
    for i in range(len(frame)):
        if i > 0 and i < len(frame) - 1:
            # df_train_input.append(pd.concat([[frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1],frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i],frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]]]), ignore_index = True)
            df_train_input = pd.concat([frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1:i+2]], ignore_index=True)
            # df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1], ignore_index = True)
            # df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i], ignore_index = True)
            # df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1], ignore_index = True)
            # df_train_input[["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)"]].iloc[len(frame) + i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1]
            # df_train_input[["T(t)", "WD(t)", "WS(t)", "P(t)"]].iloc[i*len(frame):i*len(frame) + i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]
            # df_train_input[["T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"]].iloc[i*len(frame):i*len(frame) + i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]
            # training_input[0:4,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1]).reshape((4,1))
            # training_input[4:8,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]).reshape((4,1))
            # training_input[8:12,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]).reshape((4,1))
        elif i == 0:
        
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i], ignore_index = True)
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i], ignore_index = True)
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1], ignore_index = True)
            # df_train_input[["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1]
            # df_train_input[["T(t)", "WD(t)", "WS(t)", "P(t)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]
            # df_train_input[["T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]
            # training_input[0:4,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]).reshape((4,1))
            # training_input[4:8,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]).reshape((4,1))
            # training_input[8:12,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]).reshape((4,1))
            
        else:
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1], ignore_index = True)
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i], ignore_index = True)
            df_train_input.append(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i], ignore_index = True)
            # df_train_input[["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1]
            # df_train_input[["T(t)", "WD(t)", "WS(t)", "P(t)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]
            # df_train_input[["T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"]].iloc[i] = frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]
            # training_input[0:4,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1]).reshape((4,1))
            # training_input[4:8,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]).reshape((4,1))
            # training_input[8:12,i*len(frame):i*len(frame) + i] = np.array(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]).reshape((4,1))
        # time = frame.valid_time.iloc[i]
        df_train_target[["T(t)", "WD(t)", "WS(t)", "P(t)"]].iloc[i] = df_true[['temp', 'wind_direction', 'wind_speed', 'precip_quantity_6hour']].iloc[i]
        # training_target[0:4,i*len(frame):i*len(frame) + i] = np.array(df_true[['temp', 'wind_direction', 'wind_speed', 'precip_quantity_6hour']].iloc[i]).reshape((4,1))
    # return pd.DataFrame(list(zip(oneTrainingInput, oneTarget)), columns=["Input", "Target"]) 

# training_input = pd.DataFrame(training_input.transpose(), columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])
# training_target = pd.DataFrame(training_target.transpose(), columns=["T(t)", "WD(t)", "WS(t)", "P(t)"])
