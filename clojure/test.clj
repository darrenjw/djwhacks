; test.clj
; quick test of some clojure features

(defn fib [n]
 (if (= n 0)
   1 
   (* n (fib (- n 1))
   )
  )
)

(def f5 (fib 5))
(println "5! =" f5)



; eof


