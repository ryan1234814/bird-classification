import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, Square, Upload, Sparkles, Loader2, Bird as BirdIcon, 
  MapPin, Info, Image as ImageIcon, CheckCircle, Navigation, 
  ChevronRight, Volume2, Globe, Trees, Compass
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import SpotlightCard from './components/SpotlightCard';
import AudioVisualizer from './components/AudioVisualizer';

// Fix for default marker icon in react-leaflet
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});
L.Marker.prototype.options.icon = DefaultIcon;

const API_BASE_URL = 'http://localhost:8000';

// Component to dynamically update map view when coordinates change
function ChangeView({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [recordTime, setRecordTime] = useState(0);

  // UI State
  const [activeTab, setActiveTab] = useState('detections');
  const [selectedBirdIdx, setSelectedBirdIdx] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordTime((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
      setRecordTime(0);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100,
          channelCount: 1
        }
      });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        await uploadAudio(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Microphone access denied. Please check your browser permissions.');
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const uploadAudio = async (blob) => {
    setIsLoading(true);
    setResults(null);
    const formData = new FormData();
    formData.append('file', blob, 'recording.wav');

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data);
      setSelectedBirdIdx(0);
      setActiveTab('detections');
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis engine experienced an error. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      uploadAudio(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': [] },
    multiple: false
  });

  const getActiveBird = () => {
    if (!results || !results.detections || results.detections.length === 0) return null;
    return results.detections[selectedBirdIdx];
  };

  const activeBird = getActiveBird();

  const renderTabNavigation = () => (
    <div className="tabs-container">
      <button
        className={`tab-button ${activeTab === 'detections' ? 'active' : ''}`}
        onClick={() => setActiveTab('detections')}
      >
        <Volume2 size={18} /> Detections
      </button>
      <button
        className={`tab-button ${activeTab === 'map' ? 'active' : ''} ${!activeBird ? 'opacity-50 cursor-not-allowed' : ''}`}
        onClick={() => activeBird && setActiveTab('map')}
        disabled={!activeBird}
      >
        <Globe size={18} /> Major Places
      </button>
      <button
        className={`tab-button ${activeTab === 'visual' ? 'active' : ''}`}
        onClick={() => setActiveTab('visual')}
      >
        <Sparkles size={18} /> Acoustic Data
      </button>
    </div>
  );

  return (
    <div className="min-h-screen text-slate-50">
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-12 md:py-20"
      >
        <div className="flex items-center justify-center gap-4 mb-4">
          <div className="p-3 bg-emerald-500/10 rounded-2xl border border-emerald-500/20 shadow-2xl shadow-emerald-500/10">
            <BirdIcon size={48} className="text-emerald-400" />
          </div>
          <h1>BirdScan AI</h1>
        </div>
        <p className="text-emerald-50/60 max-w-2xl mx-auto text-lg md:text-xl font-medium px-4">
          Nature intelligence at your fingertips. Identify species by sound, 
          explore global habitats, and discover conservation hotspots.
        </p>
      </motion.header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 pb-20">
        {/* SIDEBAR: Capture */}
        <div className="lg:col-span-4 space-y-6">
          <section className="glass-card">
            <div className="flex items-center gap-3 mb-8">
              <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center">
                <Mic size={20} className="text-emerald-400" />
              </div>
              <h2 className="text-xl">Capture Species</h2>
            </div>

            <div className="space-y-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={isLoading}
                  className="btn-primary w-full group overflow-hidden relative"
                >
                  <div className="absolute inset-0 bg-white/10 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500 skew-x-12" />
                  <Mic size={22} />
                  <span>Ambient Recording</span>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="btn-primary w-full !bg-rose-500 shadow-rose-500/30 glow-rose"
                >
                  <Square size={20} fill="white" />
                  <span>Stop ({recordTime}s)</span>
                </button>
              )}

              <AudioVisualizer isRecording={isRecording} />

              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'border-emerald-400 bg-emerald-400/5' : ''}`}>
                <input {...getInputProps()} />
                <div className="p-4 bg-emerald-500/5 rounded-2xl inline-block mb-4 border border-emerald-500/10">
                  <Upload size={32} className="text-emerald-400/60" />
                </div>
                <div className="text-emerald-50/80 font-bold">Import Audio File</div>
                <div className="text-emerald-50/40 text-sm mt-1">.WAV, .MP3, or .M4A accepted</div>
              </div>
            </div>

            {isLoading && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-8 p-6 bg-emerald-950/20 rounded-2xl border border-emerald-500/10 text-center"
              >
                <Loader2 className="loading-spinner mx-auto mb-3 text-emerald-400" size={32} />
                <div className="text-emerald-100 font-bold">Neural Analysis in Progress</div>
                <div className="text-emerald-50/40 text-sm mt-1">Matching acoustic signatures...</div>
              </motion.div>
            )}

            {error && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-6 p-4 bg-rose-500/10 border border-rose-500/20 rounded-2xl text-rose-200 text-sm text-center"
              >
                {error}
              </motion.div>
            )}
          </section>
        </div>

        {/* MAIN: Results */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {!results && !isLoading ? (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-card flex flex-col items-center justify-center h-full min-h-[500px] border-dashed text-center"
              >
                <div className="relative mb-8">
                  <div className="absolute inset-0 bg-emerald-500/20 blur-3xl rounded-full" />
                  <Compass size={80} className="text-emerald-400/20 relative animate-pulse" />
                </div>
                <h3 className="text-2xl font-bold text-emerald-50/60">Awaiting Signal Data</h3>
                <p className="text-emerald-50/30 max-w-xs mt-2">
                  Start a recording or upload an audio file to begin the identification process.
                </p>
              </motion.div>
            ) : results ? (
              <motion.div 
                key="results"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                {/* Spotlight for top result */}
                {results.detections.length > 0 && selectedBirdIdx === 0 && (
                  <SpotlightCard bird={results.detections[0]} />
                )}

                <div className="glass-card">
                  {renderTabNavigation()}

                  <div className="min-h-[400px]">
                    {/* TAB: DETECTIONS */}
                    {activeTab === 'detections' && (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-2"
                      >
                        <div className="grid grid-cols-[3fr_2fr_1fr] text-emerald-500/40 text-xs font-black uppercase tracking-widest px-4 mb-4">
                          <div>Species Identification</div>
                          <div>Scientific Lineage</div>
                          <div className="text-right">Match</div>
                        </div>
                        
                        {results.detections.map((bird, idx) => (
                          <div
                            key={idx}
                            onClick={() => setSelectedBirdIdx(idx)}
                            className={`bird-row group ${selectedBirdIdx === idx ? 'active' : ''}`}
                          >
                            <div className="flex items-center gap-3">
                              <div className={`w-2 h-2 rounded-full transition-all duration-300 ${selectedBirdIdx === idx ? 'bg-amber-500 scale-125' : 'bg-emerald-500/20 group-hover:bg-emerald-500/40'}`} />
                              <div className="font-bold text-lg">{bird.common_name}</div>
                            </div>
                            <div className="text-emerald-400/60 italic text-sm font-medium">{bird.scientific_name}</div>
                            <div className="text-right">
                              <span className={`confidence-chip ${bird.confidence > 0.7 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
                                {(bird.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        ))}

                        {results.detections.length === 0 && (
                          <div className="text-center py-20">
                            <BirdIcon size={48} className="mx-auto text-emerald-500/10 mb-4" />
                            <div className="text-emerald-50/40 font-bold text-lg">No Clear Matches Found</div>
                            <p className="text-emerald-50/20 text-sm">Background noise may be too high.</p>
                          </div>
                        )}
                      </motion.div>
                    )}

                    {/* TAB: MAJOR PLACES (Enhanced Map) */}
                    {activeTab === 'map' && activeBird && (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="space-y-6"
                      >
                        <div className="flex items-center gap-2 text-emerald-400 text-sm font-bold bg-emerald-500/5 p-3 rounded-xl border border-emerald-500/10">
                          <Trees size={16} />
                          Conservation Hotspots & Observed Habitats
                        </div>

                        <div className="map-container relative">
                          <MapContainer
                            center={[activeBird.description?.map_coordinates?.lat || 0, activeBird.description?.map_coordinates?.lng || 0]}
                            zoom={activeBird.description?.map_coordinates?.zoom || 3}
                            style={{ height: '100%', width: '100%' }}
                            zoomControl={false}
                          >
                            <ChangeView 
                              center={[activeBird.description?.map_coordinates?.lat || 0, activeBird.description?.map_coordinates?.lng || 0]} 
                              zoom={activeBird.description?.map_coordinates?.zoom || 3} 
                            />
                            <TileLayer
                              attribution='&copy; OpenStreetMap'
                              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                              className="map-dark-tiles"
                            />
                            
                            {/* Primary Habitat Marker */}
                            <Marker position={[activeBird.description?.map_coordinates?.lat || 0, activeBird.description?.map_coordinates?.lng || 0]}>
                              <Popup>
                                <div className="p-2">
                                  <div className="font-black text-emerald-600">{activeBird.common_name}</div>
                                  <div className="text-xs text-slate-500 leading-tight mt-1">{activeBird.description?.distribution_regions}</div>
                                </div>
                              </Popup>
                            </Marker>

                            {/* Major Places Markers */}
                            {activeBird.description?.major_places?.map((place, i) => (
                              <Marker key={i} position={[place.lat, place.lng]}>
                                <Popup>
                                  <div className="p-2 max-w-xs">
                                    <div className="font-black text-amber-600 flex items-center gap-1">
                                      <MapPin size={12} /> {place.name}
                                    </div>
                                    <div className="text-xs text-slate-500 leading-tight mt-1">{place.description}</div>
                                  </div>
                                </Popup>
                              </Marker>
                            ))}
                          </MapContainer>
                          
                          {/* Map Legend/Overlay */}
                          <div className="absolute bottom-4 left-4 z-[1000] p-3 bg-slate-950/80 backdrop-blur-md rounded-xl border border-white/10 text-[10px] space-y-1">
                            <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-[#3d85c6]" /> Primary Distribution</div>
                            <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-[#f1c232]" /> Major Observation Points</div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {activeBird.description?.major_places?.map((place, i) => (
                            <div key={i} className="p-4 bg-emerald-950/20 border border-emerald-500/10 rounded-2xl">
                              <div className="text-amber-500 font-bold flex items-center gap-2 mb-1">
                                <MapPin size={14} /> {place.name}
                              </div>
                              <p className="text-xs text-emerald-50/50 leading-relaxed line-clamp-2">{place.description}</p>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}

                    {/* TAB: VISUALIZATIONS */}
                    {activeTab === 'visual' && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-6"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-emerald-400 text-sm font-bold">
                            <Activity size={16} /> Spectral Analysis Data
                          </div>
                          <div className="text-[10px] text-emerald-50/30 uppercase tracking-widest font-black">Resolution: 44.1kHz / 16-bit</div>
                        </div>

                        {results.visualization ? (
                          <div className="relative group">
                            <div className="absolute -inset-1 bg-emerald-500/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity rounded-[2.5rem]" />
                            <img
                              src={`data:image/png;base64,${results.visualization}`}
                              alt="Acoustic Visualizations"
                              className="w-full h-auto rounded-[2rem] border border-emerald-500/20 shadow-2xl relative"
                            />
                          </div>
                        ) : (
                          <div className="text-center py-20 bg-emerald-950/10 rounded-[2rem] border border-dashed border-emerald-500/10">
                            <ImageIcon size={48} className="mx-auto text-emerald-500/10 mb-4" />
                            <p className="text-emerald-50/20">Visualization engine not available for this sample.</p>
                          </div>
                        )}

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="p-3 bg-black/20 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] text-emerald-50/40 uppercase font-black">ZCR Mean</div>
                            <div className="text-lg font-black text-emerald-400">{(results.feature_analysis?.zero_crossing_rate_mean || 0).toFixed(3)}</div>
                          </div>
                          <div className="p-3 bg-black/20 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] text-emerald-50/40 uppercase font-black">Spectral Center</div>
                            <div className="text-lg font-black text-emerald-400">{Math.round(results.feature_analysis?.spectral_centroid_mean_hz || 0)}Hz</div>
                          </div>
                          <div className="p-3 bg-black/20 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] text-emerald-50/40 uppercase font-black">RMS Energy</div>
                            <div className="text-lg font-black text-emerald-400">{(results.feature_analysis?.rms_energy_mean || 0).toFixed(3)}</div>
                          </div>
                          <div className="p-3 bg-black/20 rounded-xl border border-white/5 text-center">
                            <div className="text-[10px] text-emerald-50/40 uppercase font-black">Parameters</div>
                            <div className="text-lg font-black text-emerald-400">{(results.feature_analysis?.total_parameters_extracted || 0).toLocaleString()}</div>
                          </div>
                        </div>

                        {results.cnn_predictions && results.cnn_predictions.length > 0 && (
                          <div className="mt-8">
                            <div className="text-emerald-400 text-sm font-bold mb-4 flex items-center gap-2">
                              <Loader2 size={16} className="text-emerald-500" />
                              Custom CNN Pipeline Insights
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {results.cnn_predictions.map((pred, i) => (
                                <div key={i} className="flex justify-between items-center p-3 bg-slate-900/40 rounded-xl border border-white/5">
                                  <span className="text-sm font-bold text-emerald-50/80">{pred.species}</span>
                                  <span className="text-xs font-mono text-emerald-500">{(pred.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

export default App;
