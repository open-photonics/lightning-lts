from qick import AveragerProgram
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


class LightningControl(AveragerProgram):
    
    def initialize(self):
        cfg=self.cfg  
        
        self.declare_gen(ch=cfg["dac_0"], nqz=1) # modulator 1
        self.declare_gen(ch=cfg["dac_1"], nqz=1) # modulator 2
        self.declare_gen(ch=cfg["dac_2"], nqz=1) # modulator 3
        self.declare_gen(ch=cfg["dac_3"], nqz=1) # modulator 4
        
        self.declare_readout(ch=cfg["adc_1"][0], length=cfg["readout_length"], freq=cfg["pulse_freq"], gen_ch=cfg["dac_2"])
                    
        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["dac_2"], ro_ch=cfg["adc_1"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=cfg["dac_2"])

        self.default_pulse_registers(ch=cfg["dac_0"], freq=freq, phase=phase, gain=int(cfg["dac_gain_0"]*20000)) # multiplication
        self.default_pulse_registers(ch=cfg["dac_1"], freq=freq, phase=phase, gain=int(cfg["dac_gain_1"]*20000)) # multiplication
        self.default_pulse_registers(ch=cfg["dac_2"], freq=freq, phase=phase, gain=int(cfg["dac_gain_2"]*20000)) # multiplication
        self.default_pulse_registers(ch=cfg["dac_3"], freq=freq, phase=phase, gain=int(cfg["dac_gain_3"]*20000)) # multiplication

        self.set_pulse_registers(ch=cfg["dac_0"], style="const", length=cfg["length"])
        self.set_pulse_registers(ch=cfg["dac_1"], style="const", length=cfg["length"])
        self.set_pulse_registers(ch=cfg["dac_2"], style="const", length=cfg["length"])
        self.set_pulse_registers(ch=cfg["dac_3"], style="const", length=cfg["length"])
        
        self.synci(200)  
    
    def body(self):
        self.pulse(ch=self.cfg["dac_0"])  
        self.pulse(ch=self.cfg["dac_1"])  
        self.pulse(ch=self.cfg["dac_3"])  
        self.measure(pulse_ch=self.cfg["dac_2"], 
             adcs=[1],
             adc_trig_offset=self.cfg["adc_trig_offset"],
             wait=True,
             syncdelay=self.us2cycles(self.cfg["relax_delay"]))
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"])) # align channels and wait 1us


class LightningSignalProcessing():
    def __init__(self, soccfg, soc, LightningConfig):
        self.soccfg = soccfg
        self.soc = soc
        self.LightningConfig = LightningConfig
    
    def visualize_input(self, fittings, input_LightningConfig):
        # Create a 1x4 grid of subplots and unpack them into individual axes.
        ch_num = len(fittings)
        if ch_num == 2:
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figsize as needed.
        else:
            fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4))  # Adjust figsize as needed.

        ###################################
        # Plot on the first subplot (left-most).
        self.LightningConfig["dac_gain_0"] = input_LightningConfig["dac_gain_0"]
        self.LightningConfig["dac_gain_1"] = 1
        self.LightningConfig["dac_gain_2"] = 0
        self.LightningConfig["dac_gain_3"] = 0
        
        lightning_runtime = LightningControl(self.soccfg, self.LightningConfig)
        result_waveform = lightning_runtime.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)
 
        peaks_set = []
        floors_set = []
        max_value = fittings[0](1)
        for ii, iq in enumerate(result_waveform[0][0:]):
            light_intensity = np.abs(iq[0]+1j*iq[1])
            avg_window = self.find_plateaus(signal=light_intensity, window_size=30)
            peak, floor = self.find_peak_floor(avg_window)
            peaks_set.append(peak)
            floors_set.append(floor)
            ax1.plot(range(len(light_intensity)), [x/max_value for x in light_intensity], linewidth=1, label=ii)
        ax1.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity), linewidth=1, linestyle="-")
        ax1.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set))/(max_value)), fontsize=20)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_title('Input 0')

        ###################################
        # Plot on the second subplot.
        if (ch_num >= 2):
            self.LightningConfig["dac_gain_0"] = 1
            self.LightningConfig["dac_gain_1"] = input_LightningConfig["dac_gain_1"]
            self.LightningConfig["dac_gain_2"] = 0
            self.LightningConfig["dac_gain_3"] = 0

            lightning_runtime = LightningControl(self.soccfg, self.LightningConfig)
            result_waveform = lightning_runtime.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)

            peaks_set = []
            floors_set = []
            max_value = fittings[1](1)
            for ii, iq in enumerate(result_waveform[0][0:]):
                light_intensity = np.abs(iq[0]+1j*iq[1])
                avg_window = self.find_plateaus(signal=light_intensity, window_size=30)
                peak, floor = self.find_peak_floor(avg_window)
                peaks_set.append(peak)
                floors_set.append(floor)
                ax2.plot(range(len(light_intensity)), [x/max_value for x in light_intensity], linewidth=1, label=ii)
            ax2.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity), linewidth=1, linestyle="-")
            ax2.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set))/(max_value)), fontsize=20)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_title('Input 1')

        ###################################
        # Plot on the third subplot.
        if (ch_num >= 3):
            self.LightningConfig["dac_gain_0"] = 0
            self.LightningConfig["dac_gain_1"] = 0
            self.LightningConfig["dac_gain_2"] = input_LightningConfig["dac_gain_2"]
            self.LightningConfig["dac_gain_3"] = 1

            lightning_runtime = LightningControl(self.soccfg, self.LightningConfig)
            result_waveform = lightning_runtime.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)

            peaks_set = []
            floors_set = []
            max_value = fittings[2](1)
            for ii, iq in enumerate(result_waveform[0][0:]):
                light_intensity = np.abs(iq[0]+1j*iq[1])
                avg_window = self.find_plateaus(signal=light_intensity, window_size=30)
                peak, floor = self.find_peak_floor(avg_window)
                peaks_set.append(peak)
                floors_set.append(floor)
                ax3.plot(range(len(light_intensity)), [x/max_value for x in light_intensity], linewidth=1, label=ii)
            ax3.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity), linewidth=1, linestyle="-")
            ax3.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set))/(max_value)), fontsize=20)
            ax3.set_ylim(-0.1, 1.1)
            ax3.set_title('Input 2')

        ###################################
        # Plot on the fourth subplot (right-most).
        if (ch_num >= 4):
            self.LightningConfig["dac_gain_0"] = 0
            self.LightningConfig["dac_gain_1"] = 0
            self.LightningConfig["dac_gain_2"] = 1
            self.LightningConfig["dac_gain_3"] = input_LightningConfig["dac_gain_3"]

            lightning_runtime = LightningControl(self.soccfg, self.LightningConfig)
            result_waveform = lightning_runtime.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)

            peaks_set = []
            floors_set = []
            max_value = fittings[3](1)
            for ii, iq in enumerate(result_waveform[0][0:]):
                light_intensity = np.abs(iq[0]+1j*iq[1])
                avg_window = self.find_plateaus(signal=light_intensity, window_size=30)
                peak, floor = self.find_peak_floor(avg_window)
                peaks_set.append(peak)
                floors_set.append(floor)
                ax4.plot(range(len(light_intensity)), [x/max_value for x in light_intensity], linewidth=1, label=ii)
            ax4.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity), linewidth=1, linestyle="-")
            ax4.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set))/(max_value)), fontsize=20)
            ax4.set_ylim(-0.1, 1.1)
            ax4.set_title('Input 3')

        # Adjust the layout to prevent overlapping titles.
        plt.tight_layout()

        # Display the plots.
        plt.show()
        
    
    def get_signal_timeseries(self, waveform):
        light_intensity = np.zeros(len(waveform[0][0][0]))
        for ii, iq in enumerate(waveform[0]):
            it = np.abs(iq[0]+1j*iq[1])     
            light_intensity += it
            
        light_intensity = light_intensity/len(waveform[0]) 
        
        return light_intensity
    
    def find_plateaus(self, signal, window_size):
        avg_list = []

        for i in range(len(signal)-window_size):
            m = np.mean(signal[i:i+window_size])
            avg_list.append(m)

        return avg_list

    def find_peak_floor(self, signal):
        avg = np.mean(signal)
        maximum = np.max(signal)
        minimum = np.min(signal)
        amplitude = maximum - minimum

        if maximum - avg > avg - minimum:
            peak = maximum
        else:
            peak = minimum

        floor_list = []

        for s in signal:
            if abs(s-peak) > 0.9*amplitude:
                floor_list.append(s)
        floor = np.mean(floor_list)    

        return peak, floor

    def plot_waveform(self, iq_list, max_value, floor_value, fitting, verbose=False):
        peaks_set = []
        floors_set = []
        for ii, iq in enumerate(iq_list):
            light_intensity = np.abs(iq[0]+1j*iq[1])        
            avg_window = self.find_plateaus(signal=light_intensity, window_size=30)
            peak, floor = self.find_peak_floor(avg_window)
            peaks_set.append(peak)
            floors_set.append(floor)

        for ii, iq in enumerate(iq_list):
            light_intensity = np.abs(iq[0]+1j*iq[1])
            plt.plot(range(len(light_intensity)), [fitting((x-floor_value)/(max_value-floor_value)) for x in light_intensity], linewidth=3)
        plt.hlines(fitting((np.mean(peaks_set)-floor_value)/(max_value-floor_value)), 0, len(light_intensity))
        plt.text(70, fitting((np.mean(peaks_set)-floor_value)/(max_value-floor_value))+0.1, "peak={:.2f}".format(fitting((np.mean(peaks_set)-floor_value)/(max_value-floor_value)), fontsize=20))
        
        if verbose:
            plt.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity))
            plt.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set)-floor_value)/(max_value-floor_value)), fontsize=20)
            plt.hlines((np.mean(floors_set)-floor_value)/(max_value-floor_value), 0, len(light_intensity))
            plt.text(70, (np.mean(floors_set)-floor_value)/(max_value-floor_value)+0.1, "floor={:.3f}".format((np.mean(floors_set)-floor_value)/(max_value-floor_value)), fontsize=20)
            
        plt.ylabel("Normalized Light Intensity")
        plt.xlabel("Time (ns)")
        plt.ylim((-0.2,1.2))
        plt.tight_layout()
        plt.show()
        
    def estimate_multiply_PD_absolute(self, input_0, input_1, fitting_mod_0, fitting_mod_1):
        mod_1_output = fitting_mod_0(input_0)*fitting_mod_1(input_1)/fitting_mod_1(1)

        return mod_1_output

        
class LightningCompute(LightningSignalProcessing):
    def __init__(self, soccfg, soc, LightningConfig):
        self.soccfg = soccfg
        self.soc = soc
        self.LightningConfig = LightningConfig
        self.fitting = {}
    
    def calibration(self, target_mod, plotting=False):
        measurement = []
        ground_truth = []
        
        for i in tqdm(range(256)):
            if target_mod == 0:  # multiplication modulator pair 1
                self.LightningConfig["dac_gain_0"] = float(i/255)
                self.LightningConfig["dac_gain_1"] = 1
                self.LightningConfig["dac_gain_2"] = 0
                self.LightningConfig["dac_gain_3"] = 0
            elif target_mod == 1:  # multiplication modulator pair 1
                self.LightningConfig["dac_gain_0"] = 1
                self.LightningConfig["dac_gain_1"] = float(i/255)
                self.LightningConfig["dac_gain_2"] = 0
                self.LightningConfig["dac_gain_3"] = 0
            elif target_mod == 2:  # multiplication modulator pair 2
                self.LightningConfig["dac_gain_0"] = 0
                self.LightningConfig["dac_gain_1"] = 0
                self.LightningConfig["dac_gain_2"] = float(i/255)
                self.LightningConfig["dac_gain_3"] = 1
            elif target_mod == 3:  # multiplication modulator pair 2
                self.LightningConfig["dac_gain_0"] = 0
                self.LightningConfig["dac_gain_1"] = 0
                self.LightningConfig["dac_gain_2"] = 1
                self.LightningConfig["dac_gain_3"] = float(i/255)
            
            result = self.photonic_computing(self.LightningConfig)
            measurement.append(result)
            ground_truth.append(i/255)

        normalized_input = [x/255 for x in range(256)]
        
        fitting_to_measurement = np.poly1d(np.polyfit(normalized_input, measurement, deg=2))
        self.fitting[target_mod] = fitting_to_measurement
        
        if plotting:
            plt.figure()
            plt.scatter(normalized_input, measurement, s=50, facecolors='none', edgecolors='green')
            plt.plot(np.linspace(0, 1, num=256), fitting_to_measurement(np.linspace(0, 1, num=256)), color="black", linewidth=3, linestyle="--")
            plt.xlabel("DAC input number")
            plt.ylabel("Lightning ADC readouts")
            plt.title("Modulator Transfer function y={:.3f}x^2 + {:.3f}x + {:.3f}".format(fitting_to_measurement[2], fitting_to_measurement[1], fitting_to_measurement[0]))
            plt.tight_layout()
            plt.show()

        return fitting_to_measurement
    
    def decoder(self, sample_points, plotting=False):
        actual_received_result = []
        normalized_product = []
        
        LightningConfig["dac_gain_0"] = 1
        LightningConfig["dac_gain_1"] = 1
        max_actual_received_result = self.photonic_computing(LightningConfig)
        actual_received_result.append(max_actual_received_result)
        normalized_product.append(1)

        for x in tqdm(range(sample_points-1)):
            input_0 = random.randint(0, 255)/255
            input_1 = random.randint(0, 255)/255
            LightningConfig["dac_gain_0"] = input_0
            LightningConfig["dac_gain_1"] = input_1
            actual_received_result.append(self.photonic_computing(LightningConfig))
            normalized_product.append(input_0*input_1)
        
        normalized_actual_received_result = [x/max_actual_received_result for x in actual_received_result]

        fitting_decoder = np.polynomial.Chebyshev.fit(normalized_actual_received_result, normalized_product, deg=3, domain=[0,1])
        
        if plotting:
            plt.figure()
            plt.scatter(normalized_actual_received_result, [x for x in normalized_product], s=50, facecolors='none', edgecolors='green')
            accumu_error = 0
            for i in range(len(normalized_actual_received_result)):
                accumu_error += pow((normalized_product[i] - normalized_actual_received_result[i]), 2)
            plt.plot(np.linspace(0, 1, num=256), fitting_decoder(np.linspace(0, 1, num=256)), color="black", linewidth=3, linestyle="--")
            plt.title ("decoder accuracy is {:.3f}%".format(100-100*np.sqrt(accumu_error/len(normalized_actual_received_result))))
            plt.xlabel("Lightning actual readouts")
            plt.ylabel("normalized_compute result")

            plt.tight_layout()
            plt.show()
            
        return fitting_decoder
    
    def photonic_computing(self, LightningConfig, plotting = False):
        lightning_runtime = LightningControl(self.soccfg, self.LightningConfig)
        result_waveform = lightning_runtime.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)
        
        if plotting: 
            plt.figure()
            max_value = self.fitting[0](1) * self.fitting[1](1) + self.fitting[2](1) * self.fitting[3](1) # run this after calibration
            
        peaks_set = []
        floors_set = []
        
        for ii, iq in enumerate(result_waveform[0]):
            light_intensity = np.abs(iq[0]+1j*iq[1])        
            avg_window = self.find_plateaus(signal=light_intensity, window_size=int(LightningConfig["length"]*0.75))
            peak, floor = self.find_peak_floor(avg_window)
            peaks_set.append(peak)
            floors_set.append(floor)
            if plotting:
                plt.plot(range(len(light_intensity)), [x/max_value for x in light_intensity], linewidth=1, label=ii)

        result = abs(np.mean(peaks_set) - np.mean(floors_set))
        
        if plotting:
            plt.hlines((np.mean(peaks_set))/(max_value), 0, len(light_intensity), linewidth=1, linestyle="-")
            plt.text(70, (np.mean(peaks_set))/(max_value)+0.1, "peak={:.3f}".format((np.mean(peaks_set))/(max_value)), fontsize=20)
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.show()
        
        return result
    
    # we can estimate the PD readout numbers for multiply using this function
    def estimate_multiplication_results(self, input_0, input_1, fitting_mod_0, fitting_mod_1):
        mul_output = fitting_mod_0(input_0) * fitting_mod_1(input_1)/fitting_mod_1(1)

        return mul_output
    
    # we can estimate the PD readout numbers for multiply and accumulate using this function
    def estimate_MAC_results(self, input_0, input_1, input_2, input_3, fitting_mod_0, fitting_mod_1, fitting_mod_2, fitting_mod_3):
        add_output = self.estimate_multiplication_results(input_0, input_1, fitting_mod_0, fitting_mod_1) + self.estimate_MAC_results(input_2, input_3, fitting_mod_2, fitting_mod_3)

        return add_output
    
    
LightningConfig={
    "dac_0":0, # --Fixed
    "dac_1":1, # --Fixed
    "dac_2":2, # --Fixed
    "dac_3":3, # --Fixed
    "adc_1":[1], # --Fixed
    "reps":4, # --Fixed
    "relax_delay":1, # --us
    "res_phase":0, # --degrees
    "length":100, # [Clock ticks] # Try varying length from 10-100 clock ticks
    "readout_length":250, # [Clock ticks] # Try varying readout_length from 50-1000 clock ticks
    "dac_gain_0":1, # [DAC units] # Try varying pulse_gain from 500 to 30000 DAC units
    "dac_gain_1":1, # [DAC units] # Try varying pulse_gain from 500 to 30000 DAC units
    "dac_gain_2":1, # [DAC units] # Try varying pulse_gain from 500 to 30000 DAC units
    "dac_gain_3":1, # [DAC units] # Try varying pulse_gain from 500 to 30000 DAC units
    "pulse_freq": 0, # [MHz]
    "adc_trig_offset": 100, # [Clock ticks] # Try varying adc_trig_offset from 100 to 220 clock ticks
    "soft_avgs":100 # Try varying soft_avgs from 1 to 200 averages
    }