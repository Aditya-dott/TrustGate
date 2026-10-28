/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#08090c',
          900: '#0e0f13',
          800: '#14161c',
          700: '#1a1d25',
          600: '#262a34',
          500: '#363a45',
          400: '#4b5162',
          300: '#6b7280',
          200: '#a1a7b5',
          100: '#e2e6ee',
          50:  '#f5f7fb',
        },
        iris: {
          50:  '#f1f0fe',
          100: '#e4e1fc',
          200: '#c9c4f9',
          300: '#a99ff5',
          400: '#8d7ff2',
          500: '#7c73f0',
          600: '#6156e0',
          700: '#4c42c9',
          800: '#3f37a3',
          900: '#332d82',
        },
        mint: {
          400: '#5be4a8',
          500: '#3ecf8e',
          600: '#2fb377',
        },
        sun: {
          400: '#ffc472',
          500: '#ffb454',
          600: '#e89a3d',
        },
        ember: {
          400: '#ffa770',
          500: '#ff9558',
          600: '#e87a3f',
        },
        crimson: {
          400: '#fa7169',
          500: '#f85149',
          600: '#dc3d36',
        },
        sky: {
          400: '#7fc6ff',
          500: '#5eb8ff',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      fontVariantNumeric: {
        tabular: 'tabular-nums',
      },
      boxShadow: {
        'glow-iris': '0 0 0 1px rgba(124,115,240,0.35), 0 8px 32px -8px rgba(124,115,240,0.35)',
        'glow-mint': '0 0 0 1px rgba(62,207,142,0.35), 0 8px 24px -8px rgba(62,207,142,0.3)',
        'glow-crimson': '0 0 0 1px rgba(248,81,73,0.35), 0 8px 24px -8px rgba(248,81,73,0.3)',
        'panel': '0 1px 0 0 rgba(255,255,255,0.03) inset, 0 24px 48px -24px rgba(0,0,0,0.6)',
      },
      backgroundImage: {
        'grid-fade':
          'radial-gradient(circle at 50% 0%, rgba(124,115,240,0.10), transparent 55%), linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0) 60%)',
      },
      animation: {
        'fade-in': 'fadeIn 250ms ease-out',
        'slide-up': 'slideUp 300ms cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-soft': 'pulseSoft 2.4s ease-in-out infinite',
        'shimmer': 'shimmer 2.4s linear infinite',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: {
          '0%': { transform: 'translateY(6px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        pulseSoft: {
          '0%, 100%': { opacity: 0.5 },
          '50%': { opacity: 1 },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
