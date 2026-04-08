import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Award, Star, Search, MapPin, Feather, Wind, Activity } from 'lucide-react';
import confetti from 'canvas-confetti';

const SpotlightCard = ({ bird }) => {
  useEffect(() => {
    if (bird && bird.confidence > 0.8) {
      const duration = 3 * 1000;
      const animationEnd = Date.now() + duration;
      const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };

      const randomInRange = (min, max) => Math.random() * (max - min) + min;

      const interval = setInterval(function() {
        const timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
          return clearInterval(interval);
        }

        const particleCount = 50 * (timeLeft / duration);
        confetti({ ...defaults, particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } });
        confetti({ ...defaults, particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } });
      }, 250);

      return () => clearInterval(interval);
    }
  }, [bird]);

  if (!bird) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6, type: "spring" }}
      className="glass-card relative overflow-hidden border-2 border-emerald-500/30 mb-8"
    >
      {/* Decorative Gradient Background */}
      <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 -ml-16 -mb-16 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
          <div>
            <div className="flex items-center gap-2 text-amber-500 font-bold text-sm uppercase tracking-widest mb-2">
              <Award size={18} />
              Top Match Identified
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-white leading-tight">
              {bird.common_name}
            </h2>
            <div className="text-emerald-400 italic text-xl mt-1 font-medium">
              {bird.scientific_name}
            </div>
          </div>

          <div className="flex flex-col items-end">
            <div className="text-xs text-emerald-500/60 uppercase font-black tracking-tighter mb-1">Match Confidence</div>
            <div className="flex items-baseline gap-1">
              <span className="text-5xl font-black text-emerald-400">{(bird.confidence * 100).toFixed(1)}</span>
              <span className="text-2xl font-bold text-emerald-500/60">%</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="info-card !bg-emerald-950/20 border-emerald-500/20">
              <div className="info-label"><Feather size={14} /> Characteristics</div>
              <p className="text-emerald-50/80 leading-relaxed">
                {bird.description?.physical_characteristics || "Species specific details not available."}
              </p>
            </div>
            
            <div className="info-card !bg-amber-950/20 border-amber-500/20">
              <div className="info-label !text-amber-500"><Search size={14} /> Key Locations</div>
              <div className="flex flex-wrap gap-2 mt-2">
                {bird.description?.major_places?.slice(0, 3).map((place, i) => (
                  <span key={i} className="px-3 py-1 bg-amber-500/10 border border-amber-500/20 rounded-full text-amber-200 text-xs font-bold">
                    {place.name}
                  </span>
                )) || <span className="text-amber-200/50 italic text-sm">Location data not available</span>}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="info-card !bg-slate-900/40">
              <div className="info-label"><Wind size={14} /> Habitat</div>
              <p className="text-emerald-50/80 leading-relaxed">
                {bird.description?.habitat || "Habitat data not available."}
              </p>
            </div>
            
            <div className="info-card !bg-slate-900/40">
              <div className="info-label"><Activity size={14} /> Diet</div>
              <p className="text-emerald-50/80 leading-relaxed">
                {bird.description?.diet || "Dietary data not available."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default SpotlightCard;
