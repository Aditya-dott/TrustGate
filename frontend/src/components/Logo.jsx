// Custom logo — a shield with an inner bar mark. Uses a soft iris → sky
// gradient. Size prop controls the pixel size; defaults to 28.

export default function Logo({ size = 28, animate = true }) {
  const gid = `tg-grad-${size}`
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="flex-shrink-0"
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#8d7ff2" />
          <stop offset="1" stopColor="#5eb8ff" />
        </linearGradient>
      </defs>
      <path
        d="M16 3l11 4v9c0 7.5-5.2 13.9-11 15-5.8-1.1-11-7.5-11-15V7l11-4z"
        fill={`url(#${gid})`}
      />
      <path d="M13 11h6v6h-6v-6zm0 8h6v3h-6v-3z" fill="#08090c" />
      {animate && (
        <circle
          cx="16"
          cy="16"
          r="14"
          fill="none"
          stroke={`url(#${gid})`}
          strokeOpacity="0.35"
          strokeWidth="0.8"
          className="animate-pulse-soft"
        />
      )}
    </svg>
  )
}
