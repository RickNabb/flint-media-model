;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; COGNITIVE CONTAGION MODEL
;; Author: Nick Rabb (nicholas.rabb@tufts.edu)
;;
;; This simulation space contains code to generate several different contagion models
;; across different network topologies and propagation types. It implements the simple,
;; complex, and cognitive contagion models; several contagion functions; and various
;; parameterizations of graph topologies to test against.
;;
;; It also has implementations of different message set capabilities, read from external files,
;; to run consistent message simulations. Moreover, graphs can be exported to text files that can
;; be read in to ensure consistency across simulations.
;;
;; This simulation frequently interfaces with external Python scripts, so there are a handful of
;; helper functions in the project to assist with data conversion & parsing. There are also functions
;; useful for post-simulation data analysis, as well as BehaviorSpace pipelines to run experiments.
;;
;; TODO:
;; - Move agent memory to numpy data structure for speed
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

extensions [py]

globals [
  ;; Messaging
  selected-turtle
  mag-g
  cur-message-id
  messages-over-time
  citizen-priors
  citizen-malleables

  ;; Media ecosystem
  communities

  ;; For experiments
  contagion-dir
  behavior-rand

  ;; Agents who believed at t-1
  num-agents-adopted
  agents-adopted-by-tick
]

citizens-own [
  ;; Messaging
  brain
  messages-heard
  messages-believed
  is-flint?
]

medias-own [
  idee
  brain
  messages-heard
  messages-believed
]

breed [ medias media ]
breed [ citizens citizen ]

directed-link-breed [ social-friends social-friend ]
directed-link-breed [ subscribers subscriber ]
directed-link-breed [ media-peers media-peer ]

social-friends-own [ weight ]
subscribers-own [ weight ]

;;;;;;;;;;;;;;;;;
;; SETUP PROCS
;;;;;;;;;;;;;;;;;

to setup
  clear-all
  set-default-shape turtles "circle"
  set-default-shape medias "box"

  ;; Python imports and setup
  setup-py

  ;; Set the priors and malleables for each citizen
  set citizen-priors []
  set citizen-malleables [ "Attributes.A" ]

  set agents-adopted-by-tick []

  ask patches [
    set pcolor white
  ]

  ifelse not load-graph? [
    create-citizenz
    connect-agents
    set communities communities-by-level
    create-flint-citizens
    create-media
    connect-media
  ] [
    read-graph
  ]

  ;; Layout turtles
  let max_turtle max-one-of turtles [ count social-friend-neighbors ]
  if graph-type = "erdos-renyi" [
    ask turtles with [ count social-friend-neighbors = 0 ] [ setxy random-xcor random-ycor ]
    repeat 120 [ layout-spring turtles social-friends 0.3 10 2 ]
  ]
  if graph-type = "mag" [
    repeat 120 [ layout-spring turtles social-friends 0.6 10 10 ]
  ]
  if graph-type = "watts-strogatz" [
    layout-circle sort citizens 12
    repeat 2 [ layout-spring citizens social-friends 0.3 10 1 ]
  ]
  if graph-type = "barabasi-albert" or graph-type = "kronecker" [
    layout-radial citizens social-friends max_turtle
    layout-spring citizens social-friends 0.3 10 1
  ]

  layout

  reset-ticks
end

;; Set up all the relevant Python scripts
to setup-py
  py:setup "python"
  py:run "import sys"
  py:run "import os"
  py:run "import kronecker as kron"
  py:run "from data import *"
  py:run "from messaging import *"
  py:run "import mag as MAG"
  py:run "from nlogo_graphs import *"
end

to create-agents
  create-citizenz
  create-media
end

to create-citizen-dist [ id ]
  let prior-vals []
  let malleable-vals []
;  let prior-vals (map sample-attr-dist citizen-priors)
;  let malleable-vals (map sample-attr-dist citizen-malleables)
  create-citizen id prior-vals malleable-vals
end

to create-citizen [ id prior-vals malleable-vals ]
  let b create-agent-brain id [] [] [] []
  create-citizens 1 [
    set brain b
    set messages-heard []
    set messages-believed []
    set is-flint? false

;    set size 0.5
    set size 1
    setxy random-xcor random-ycor
  ]
end

to create-flint-citizens
  let community (flint-community communities (n * flint-community-size))
  foreach community [ cit-id ->
    ask citizen cit-id [
      set is-flint? true
      let prior-vals (item 0 (normal-dist-multiple (belief-resolution - 1) 3 1 1 (length citizen-priors)))
;      let prior-vals (map sample-attr-dist citizen-priors)
;      let malleable-vals (map sample-attr-dist citizen-malleables)
      let malleable-vals (item 0 (normal-dist-multiple (belief-resolution - 1) 3 1 1 (length citizen-malleables)))
      set brain create-agent-brain cit-id citizen-priors citizen-malleables prior-vals malleable-vals
    ]
  ]
end

to create-citizenz
  let id 0
  let en 0
  set en N
  repeat en [
    create-citizen-dist id
    set id id + 1
  ]
end

to create-media
  if media-agents? [
    let id N
    let level-sizes community-sizes-by-level communities
    foreach level-sizes [ level ->
      repeat (level + 1) [
        ; TODO: Here, this is getting rid of media that then are needed for the connect-media function...
        ; Somehow, need to mark some media as deleted, or check if they exist before connecting
;        let roll random-float 1
;        if roll <= media-connection-prob [
          create-medias 1 [
            set brain create-agent-brain id [] [] [] []
            set cur-message-id 0
            setxy random-xcor random-ycor
            set color green
            set messages-heard []
            set messages-believed []
          ]
          set id id + 1
        ]
;      ]
    ]
  ]
end

;; Connect the agents in the simulation based on the graph type selected.
to connect-agents
  let G -1
  if graph-type = "erdos-renyi" [
    set G er-graph N erdos-renyi-p
  ]
  if graph-type = "watts-strogatz" [
    set G ws-graph N watts-strogatz-k watts-strogatz-p
  ]
  if graph-type = "barabasi-albert" [
    set G ba-graph N ba-m
  ]
  if graph-type = "mag" [
    set G mag N (list-as-py-array (sort citizen-malleables) false) mag-style
;    show [dict-value brain "A"] of citizens
    let i 0

    ;; Here, we have to reset the agent beliefs to be what they were in the MAG algorithm, otherwise
    ;; the edge connections don't make sense.
    repeat length (dict-value G "L") [
      let cit-attrs (item i (dict-value G "L"))
      let j 0

      ;; We update beliefs in the sorted order of malleables then priors because that's how L is
      ;; generated in the MAG algorithm.
      foreach (sort citizen-malleables) [ attr ->
        ask citizens with [(dict-value brain "ID") = i] [
          set brain update-agent-belief brain attr (item j cit-attrs)
        ]
        set j j + 1
      ]
      foreach (sort citizen-priors) [ attr ->
        ask citizens with [(dict-value brain "ID") = i] [
          set brain update-agent-belief brain attr (item j cit-attrs)
        ]
        set j j + 1
      ]
      set i i + 1
    ]
;    show [sort (list (dict-value brain "ID") (dict-value brain "A"))] of citizens
  ]

  ; Create links
  let edges (dict-value G "edges")
  foreach edges [ ed ->
    let end-1 (item 0 ed)
    let end-2 (item 1 ed)
    let cit1 citizen end-1
    let cit2 citizen end-2
;    show (word "Linking " cit1 "(" (dict-value [brain] of cit1 "A") ") and " cit2 "(" (dict-value [brain] of cit2 "A") ")")
    ask citizen end-1 [ create-social-friend-to citizen end-2 [ set weight citizen-citizen-influence ] ]
  ]

  ; Remove isolates
  ask citizens with [ empty? sort social-friend-neighbors ] [ die ]
end

to connect-media
  let level-sizes community-sizes-by-level communities
  let media-id-base n
  let i 0
  foreach communities [ level ->
    foreach level [ cit-media-pair ->
      let media-id media-id-base + (item 1 cit-media-pair)
;      show (word "creating ties from media " media-id " to " (item 0 cit-media-pair) " with media base " media-id-base)

      ask citizen (read-from-string (item 0 cit-media-pair)) [
        create-subscriber-from (media media-id) [ set weight media-citizen-influence ]
        create-subscriber-to (media media-id) [ set weight citizen-media-influence ]
      ]
    ]

    if i < length level-sizes [
      set media-id-base media-id-base + (item i level-sizes)
      set i i + 1
    ]
  ]

  if media-monitor-peers? [
    let r N
    let peers media-peer-connections
    foreach peers [ connections ->
      let c N
      foreach connections [ connect? ->
        if connect? = 1 [
          ask media r [
            create-media-peer-to (media c)
            create-media-peer-from (media c)
          ]
        ]
        set c c + 1
      ]
      set r r + 1
    ]
  ]
;  ask medias [
;    set color scale-color green (length sort subscriber-neighbors) 100 0
;  ]
end

;;;;;;;;;;;;;;;;;
;; BEHAVIORSPACE SIMULATION
;; PROCS
;;;;;;;;;;;;;;;;;

;; For runs of the cognitive contagion simulations, set function parameters according to the type of
;; function being used.
;; NOTE: These are only used with BehaviorSpace simulations!!
to set-cognitive-contagion-params
  if cognitive-fn = "linear-gullible" [
    set cognitive-scalar? true
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-scalar 0
    set cognitive-translate 1
  ]
  if cognitive-fn = "linear-mid" [
    set cognitive-scalar? true
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-scalar 1
    set cognitive-translate 1
  ]
  if cognitive-fn = "linear-stubborn" [
    set cognitive-scalar? true
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-translate 10
    set cognitive-scalar 20
  ]
  if cognitive-fn = "threshold-gullible" [
    set cognitive-scalar? false
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-translate 6
  ]
  if cognitive-fn = "threshold-mid" [
    set cognitive-scalar? false
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-translate 3
  ]
  if cognitive-fn = "threshold-stubborn" [
    set cognitive-scalar? false
    set cognitive-exponent? false
    set cognitive-translate? true
    set cognitive-translate 1
  ]
  ;; Threshold t
  let t 1
  if cognitive-fn = "sigmoid-gullible" [
    set t belief-resolution
    set cognitive-scalar? false
    set cognitive-exponent? true
    set cognitive-translate? true
    set cognitive-exponent 1
    set cognitive-translate t + 1
  ]
  if cognitive-fn = "sigmoid-mid" [
    set t (belief-resolution * 2 / 7)
    set cognitive-scalar? false
    set cognitive-exponent? true
    set cognitive-translate? true
    set cognitive-exponent 2
    set cognitive-translate t + 1
  ]
  if cognitive-fn = "sigmoid-stubborn" [
    set t ceiling (belief-resolution / 7)
    set cognitive-scalar? false
    set cognitive-exponent? true
    set cognitive-translate? true
    set cognitive-exponent 4
    set cognitive-translate t + 1
  ]
end

to set-cit-media-over-time
  set citizen-media-influence (cit-media-gradual-scalar * ((ticks + 1) / tick-end))
end

;; Calculate the distances between the brain of cit and its neighbors.
;;
;; @param cit - The citizen to calculate the distance for.
;; @reports A list of distances between agent brains of cit and its neighbors.
to-report neighbor-distance [ cit ]
  let cit-distances []
  ask cit [
    let ego-brain brain
    ask social-friend-neighbors [
      set cit-distances lput (dist-between-agent-brains brain ego-brain) cit-distances
    ]
  ]
  report cit-distances
end

;; Calculate the average of all average citizen distances. This can be
;; used as a makeshift measure of homophily in the graph.
;;
;; @reports The average citizen distance across the entire graph.
to-report avg-citizen-distance
  let distance-n []
  ask citizens [
    let cit-distances neighbor-distance self

    ;; Some nodes may be disconnected
    if length cit-distances > 0 [
      set distance-n lput (mean cit-distances) distance-n
    ]
  ]
  report list (mean distance-n) (variance distance-n)
end

;;;;;;;;;;;;;;;;;
;; SIMULATION PROCS
;;;;;;;;;;;;;;;;;

to go
  ifelse ticks <= tick-end [
    step
  ] [ stop ]
end

to step
;  if (ticks mod 5) = 0 [
;    set num-agents-adopted 0
;  ]
;  show (word "last num agents adopted: " num-agents-adopted)
  set num-agents-adopted 0
  if cit-media-gradual? [ set-cit-media-over-time ]
  if flint-organizing? [
    ask citizens with [is-flint?] [ organize self ]
  ]
  if contagion-on? [
    ;; In the case where we do not have influencer agents, simply do a contagion from the agent perspective
    ask turtles with [ not is-agent-brain-empty? self ] [
      let c self
      ask out-link-neighbors [
;        show (word self " receiving message " (agent-brain-malleable-values c) " from " c)
        receive-message self c (agent-brain-malleable-values c) 0
      ]
    ]
  ]

  layout

  tick
end

to update-agents
  ask citizens [
    update-citizen
  ]
  ask medias [
    update-citizen
  ]
end

;; Update any display properties of agents
to update-citizen
  if show-citizen-political? [ give-self-ip-color ]
end

;; TODO: Remove this maybe?? If we don't need it
;;
;; Initiate the sending of a message from influencer agent m to its subscribers.
;;
;; @param m - The influencer agent to send the message.
;; @param message - The message to send.
;to send-media-message-to-subscribers [ m message ]
;  ask m [
;    let mid cur-message-id
;;    set messages-sent (lput (list mid message) messages-sent)
;    ask my-subscribers [
;      ask other-end [
;        receive-message self m message mid
;      ]
;    ]
;    set cur-message-id (cur-message-id + 1)
;  ]
;end

to-report set-merge-lists [ lists ]
  let merged []
  foreach lists [ l ->
    foreach l [ element -> set merged (lput element merged) ]
  ]
  set merged remove-duplicates merged
  report merged
end

to organize [ cit ]
  let max-media-subscriber-count max [ count subscribers ] of medias
  let max-social-neighbor-count max [ count social-friend-neighbors ] of citizens
  let num-cit-neighbors [ count out-link-neighbors ] of cit

  if flint-organizing-strategy = "neighbors-of-neighbors" [
    let neighbor-neighbors [ sort social-friend-neighbors ] of [ social-friend-neighbors ] of cit
    let neighbor-neighbors-merged set-merge-lists neighbor-neighbors
;    foreach neighbor-neighbors [ neighbor-set ->
;      foreach neighbor-set [ neighbor -> set neighbor-neighbors-merged (lput neighbor neighbor-neighbors-merged) ]
;    ]
;    set neighbor-neighbors-merged remove-duplicates neighbor-neighbors-merged

    foreach up-to-n-of organizing-capacity neighbor-neighbors-merged [ neighbor ->
      ask neighbor [
        if cit != self [
          let num-neighbors count out-link-neighbors
          let connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * citizen-citizen-influence
          let roll random-float 1
          let neighbor-neighbor self
          if roll <= connect-prob [
            create-social-friend-to cit
            ask cit [ create-social-friend-to neighbor-neighbor ]
          ]
        ]
      ]
    ]
  ]
  if flint-organizing-strategy = "high-degree-media" [
    let sorted-media sort-on [ count subscriber-neighbors ] (medias with [ not is-link? (out-link-to cit) ])
    let top-n []
    ifelse length sorted-media > organizing-capacity [
      set top-n sublist sorted-media (length sorted-media - organizing-capacity) (length sorted-media)
    ] [
      set top-n sorted-media
    ]
    foreach top-n [ m ->
      ask m [
        if cit != self [
          let num-neighbors count subscriber-neighbors
          let connect-prob 1
          ifelse dynamic-cit-media-influence? [
            set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * (num-cit-neighbors / (max [ length sort social-friend-neighbors ] of citizens))
          ] [
            set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * citizen-media-influence
          ]
          let roll random-float 1
          let neighbor-neighbor self
          if roll <= connect-prob [
            show (word "Connected to " cit)
            create-subscriber-to cit
            create-subscriber-from cit
          ]
        ]
      ]
    ]
  ]
  if flint-organizing-strategy = "high-degree-citizens" [
    let sorted-citizens sort-on [ count out-link-neighbors ] (citizens with [ not is-link? (out-link-to cit) ])
    let top-n sublist sorted-citizens (length sorted-citizens - organizing-capacity) (length sorted-citizens)
    foreach top-n [ m ->
      ask m [
        if cit != self [
          let num-neighbors count out-link-neighbors
          let connect-prob 1
          ; TODO: Maybe change this... should cit-cit-influence act as some sort of global measure of social trust?
          ; In that case, it would operate in concert with the dynamic trust based on degree
          ifelse dynamic-cit-cit-influence? [
            set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * (num-cit-neighbors / (max [ length sort social-friend-neighbors ] of citizens))
          ] [
            set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * citizen-citizen-influence
          ]
          let roll random-float 1
          let neighbor-neighbor self
          if roll <= connect-prob [
            show (word "Connected to " cit)
            create-social-friend-to cit
            create-social-friend-from cit
          ]
        ]
      ]
    ]
  ]
  if flint-organizing-strategy = "high-degree-cit-and-media" [
    let sorted-cit-media sort-on [ count out-link-neighbors ] (turtles with [ not is-link? (out-link-to cit) ])
    let top-n sublist sorted-cit-media (length sorted-cit-media - organizing-capacity) (length sorted-cit-media)
    foreach top-n [ m ->
      ask m [
        if cit != self [
          let num-neighbors 0
          (ifelse is-media? m [
            set num-neighbors count subscriber-neighbors
          ] is-citizen? m [
            set num-neighbors count out-link-neighbors
          ])
          let connect-prob 1

          if is-media? m [
            ifelse dynamic-cit-media-influence? [
              set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * (num-cit-neighbors / (max [ length sort social-friend-neighbors ] of citizens))
            ] [
              set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * citizen-media-influence
            ]
          ]

          if is-citizen? m [
            ifelse dynamic-cit-cit-influence? [
              set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * (num-cit-neighbors / (max [ length sort social-friend-neighbors ] of citizens))
            ] [
              set connect-prob (num-cit-neighbors / (num-cit-neighbors + num-neighbors)) * citizen-citizen-influence
            ]
          ]

          let roll random-float 1
          let neighbor-neighbor self
          if roll <= connect-prob [
            show (word "Connected to " cit)
            create-social-friend-to cit
            create-social-friend-from cit
          ]
        ]
      ]
    ]
  ]
end

to-report agent-type-influence [ p sender receiver ]
  if is-citizen? sender and is-citizen? receiver [
    ifelse dynamic-cit-cit-influence? [
      set p p * (([ length sort social-friend-neighbors ] of sender) / (max [ length sort social-friend-neighbors ] of citizens))
    ] [
      set p p * citizen-citizen-influence
    ]
  ]
  if is-media? sender and is-citizen? receiver [ set p p * media-citizen-influence ]
  if is-citizen? sender and is-media? receiver [
    ifelse dynamic-cit-media-influence? [
      set p p * (([ length sort social-friend-neighbors ] of sender) / (max [ length sort social-friend-neighbors ] of citizens))
    ] [
      set p p * citizen-media-influence
    ]
  ]
  if is-media? sender and is-media? receiver [ set p p * media-media-influence * (([ length sort subscribers ] of sender) / (max [ length sort subscribers ] of medias)) ]
  report p
end

;; Have a citizen agent receive a message: hear it, either believe it or not, and subsequently either
;; share it or not.
;;
;; @param cit - The citizen agent who is receiving the message.
;; @param sender - The originator of the message
;; @param message - The message itself.
;; @param message-id - The unique ID of the message (used so the citizen agent does not duplicate shares)
to receive-message [ cit sender message message-id ]
  ask cit [
    if not (heard-message? self ticks message-id) [
      hear-message self message-id message

      if spread-type = "cognitive" [
        let p 0
        let scalar 1
        let expon 1
        let trans 0
        let dist (dist-to-agent-brain brain message)
;        show (word "distance to brain " dist)

        if cognitive-scalar? [ set scalar cognitive-scalar ]
        if cognitive-exponent? [ set expon cognitive-exponent ]
        if cognitive-translate? [ set trans cognitive-translate ]

        ;; Good values for linear:
        if member? "linear" cognitive-fn [ set p 1 / (trans + (scalar * dist) ^ expon) ]

        ;; Good values for sigmoid: expon = -4, trans = -5 (works like old threshold function)
        if member? "sigmoid" cognitive-fn [ set p (1 / (1 + (exp (expon * (dist - trans))))) ]
;        show (word "dist: " dist)
;        show (word self ": " (dict-value brain "A") " " message " (p=" p ")")

        if member? "threshold" cognitive-fn [
          ifelse dist <= trans [ set p 1 ] [ set p 0 ]
        ]

        ;; Whether or not to believe the message
        let roll random-float 1
        set p agent-type-influence p sender cit

        if roll <= p [
;          show (word "believed with p" p " and roll " roll)
          let b brain
          set brain (believe-message-py brain message)
          believe-message self message-id message

          ; Return [-1 -1] if both are not already present
          let beliefs-from-message (map [ attr -> (list attr (dict-value b attr)) ] (map [ bel -> item 0 bel ] message))
          let non-empty-message filter [ bel -> (dict-value beliefs-from-message (item 0 bel)) != -1 ] message
          ifelse not empty? non-empty-message [
            ask out-link-neighbors [
              receive-message self cit non-empty-message message-id
            ]
          ] [
            set num-agents-adopted num-agents-adopted + 1
          ]
        ]
        update-citizen
      ]

      if spread-type = "simple" [
        let roll random-float 1
        let p agent-type-influence simple-spread-chance sender cit

        if roll <= p [
;          show(word "believing " message-id)
          ;show (believe-message-py brain message)
          let b brain
          set brain (believe-message-py brain message)
          believe-message self message-id message

          ; Return [-1 -1] if both are not already present
          let beliefs-from-message (map [ attr -> (list attr (dict-value b attr)) ] (map [ bel -> item 0 bel ] message))
          let non-empty-message filter [ bel -> (dict-value beliefs-from-message (item 0 bel)) != -1 ] message
          ifelse not empty? non-empty-message [
            ask out-link-neighbors [
              receive-message self cit non-empty-message message-id
            ]
          ] [
;            show (word "believed message from brain " (dict-value b "A") " to " (dict-value brain "A"))
            set num-agents-adopted num-agents-adopted + 1
            let agents-adopted-at-tick (dict-value agents-adopted-by-tick ticks)
            ifelse agents-adopted-at-tick = -1 [
              let adopter-sender-pair (list self sender)
              let adoptions-at-tick (list ticks (list adopter-sender-pair))
              set agents-adopted-by-tick (lput adoptions-at-tick agents-adopted-by-tick)
;              show (word "set agents-adopted-by-tick new entry" agents-adopted-by-tick)
            ] [
              let adopter-sender-pair (list self sender)
              let new-agents-adopted-at-tick (list ticks (lput adopter-sender-pair agents-adopted-at-tick))
              set agents-adopted-by-tick (replace-dict-item agents-adopted-by-tick ticks new-agents-adopted-at-tick)
;              show (word "added agents-adopted-by-tick entry" agents-adopted-by-tick)
            ]
          ]
        ]
      ]

      if spread-type = "complex" [
        let believing-neighbors 0
        ask social-friend-neighbors [
          let believes true
          foreach message [ m ->
            let attr (item 0 m)
            let val (item 1 m)
            set believes (believes and (dict-value brain attr = val))
          ]
          if believes [
            set believing-neighbors believing-neighbors + 1
          ]
        ]
;        show (word "Citizen " cit "has ratio " (believing-neighbors / length sort social-friend-neighbors))
        if (believing-neighbors / length sort out-link-neighbors) >= complex-spread-ratio [
;          show (word "Citizen " cit " believing with ratio " (believing-neighbors / length sort social-friend-neighbors))
          let b brain
          set brain (believe-message-py brain message)
          believe-message self message-id message

          ; Return [-1 -1] if both are not already present
          let beliefs-from-message (map [ attr -> (list attr (dict-value b attr)) ] (map [ bel -> item 0 bel ] message))
          let non-empty-message filter [ bel -> (dict-value beliefs-from-message (item 0 bel)) != -1 ] message
          ifelse not empty? non-empty-message [
            ask out-link-neighbors [
              receive-message self cit non-empty-message message-id
            ]
          ] [
            set num-agents-adopted num-agents-adopted + 1
          ]
        ]
      ]
    ]
  ]
end

;; Have a citizen agent believe a message and update its cognitive model. This also records
;; that the agent believed message message-id at the current tick.
;;
;; @param cit - The citizen to believe the message.
;; @param message-id - The id of the message.
;; @param message - The message itself.
to believe-message [ cit message-id message ]
  ask cit [
    let i (index-of-dict-entry messages-believed ticks)
    ifelse i != -1 [
      let messages-at-tick (item i messages-believed)
      let message-ids (item 1 messages-at-tick)
      set message-ids (lput message-id message-ids)
      set messages-at-tick (replace-item 1 messages-at-tick message-ids)
      set messages-believed (replace-item i messages-believed messages-at-tick)
    ] [
      set messages-believed (lput (list ticks (list message-id)) messages-believed)
    ]
  ]
end

;; Return whether or not a citizen agent has already heard a given message id
;; at tick tix.
;;
;; @param cit - The citizen to check.
;; @param tix - The tick number to check against.
;; @param message-id - The message ID to check for.
;; @reports a boolean whether or not the citizen has already heard that message at tick tix.
to-report heard-message? [ cit tix message-id ]
  let heard? false
  ask cit [
    let messages-at-tick (dict-value messages-heard tix)
    if messages-at-tick != -1 [
      set heard? (member? message-id messages-at-tick)
    ]
  ]
  report heard?
end

;; Have a citizen agent hear a message and record that they've done so (so they don't
;; interact with the same message again).
;;
;; @param cit - The citizen to hear the message.
;; @param message-id - The ID of the message to record with the current tick.
;; @param message - The message itself.
to hear-message [ cit message-id message ]
  ask cit [
    let i (index-of-dict-entry messages-heard ticks)
    ifelse i != -1 [
      let messages-at-tick (item i messages-heard)
      let message-ids (item 1 messages-at-tick)
      set message-ids (lput message-id message-ids)
      set messages-at-tick (replace-item 1 messages-at-tick message-ids)
      set messages-heard (replace-item i messages-heard messages-at-tick)
    ] [
      set messages-heard (lput (list ticks (list message-id)) messages-heard )
    ]
  ]
end

to test
  py:setup "python"
  py:run "from messaging import *"
end

;;;;;;;;;;;;;
; I/O PROCEDURES
;;;;;;;;;;;;;

;; Save the current state of the graph to a file (whose path is specified in the simulation interface).
;; This will save all the agents with their state variables, the social connections between citizen agents,
;; and the influencer agents state and subscribers.
to save-graph
  ;; TODO: Find some way to get the prior & malleable attributes into a list rather than hardcoding
  let cit-ip []
  let cit-social []
  foreach sort citizens [ cit ->
    set cit-ip (lput (list cit (dict-value ([brain] of cit) "A") (dict-value ([brain] of cit) "ID")) cit-ip)
    foreach sort [ out-social-friend-neighbors ] of cit [ nb ->
      set cit-social (lput (list cit nb) cit-social)
    ]
  ]
  let media-ip []
  let media-sub []
  foreach sort medias [ m ->
    set media-ip (lput (list m (dict-value ([brain] of m) "A")) media-ip)
    foreach sort [ subscriber-neighbors ] of m [ sub ->
      set media-sub (lput (list m sub) media-sub)
    ]
  ]
  let flint-citizens list-as-py-array ([ (dict-value brain "ID") ] of citizens with [is-flint?]) false
  py:run (word "save_graph('" save-graph-path "','" cit-ip "','" cit-social "','" media-ip "','" media-sub "'," flint-citizens ")")
end

;; Read a graph back in from a data file (specified by the load-graph-path variable in the interface) and
;; construct the model appropriately.
to read-graph
  let graph py:runresult(word "read_graph('" load-graph-path "')")
  let citizenz item 0 graph
  let citizens-conns item 1 graph
  let mediaz item 2 graph
  let media-subs item 3 graph

  ;; id, a
  ;; TODO: Change this from being hard-coded for one belief "A" to being general
  create-citizens (length citizenz) [
    let i read-from-string (substring (word self) 9 ((length (word self)) - 1))
    let c (item i citizenz)
    let id item 0 c
    let a read-from-string (item 1 c)
    set is-flint? read-from-string (item 2 c)

    ifelse a = -1 [
      set brain create-agent-brain id citizen-priors [] [] []
    ] [
      set brain create-agent-brain id citizen-priors citizen-malleables [] (list a)
    ]
    set messages-heard []
    set messages-believed []

    set size 0.5
    setxy random-xcor random-ycor
    set i i + 1
  ]

  create-medias (length mediaz) [
    let i (read-from-string (substring (word self) 7 ((length (word self)) - 1))) - N
    let m (item i mediaz)
    let id item 0 m
    let a read-from-string (item 1 m)

    ifelse a = -1 [
      set brain create-agent-brain id citizen-priors [] [] []
    ] [
      set brain create-agent-brain id citizen-priors citizen-malleables [] (list a)
    ]

    set messages-heard []
    set messages-believed []

    setxy random-xcor random-ycor
    set color green
    set i i + 1
  ]
  show citizens-conns

  foreach citizens-conns [ c ->
    let c1 read-from-string (item 0 c)
    let c2 read-from-string (item 1 c)
    ask citizen c1 [ create-social-friend-to citizen c2 [ set weight citizen-citizen-influence ] ]
  ]

  foreach media-subs [ sub ->
    let c read-from-string (item 0 sub)
    let m read-from-string (item 1 sub)
    ask media m [
      create-subscriber-to citizen c [ set weight media-citizen-influence ]
      create-subscriber-from citizen c [ set weight citizen-media-influence ]
    ]
  ]
end

;;;;;;;;;;;;;
; DISPLAY PROCEDURES
;;;;;;;;;;;;;

to make-link-transparent
  ifelse is-list? color [
    ifelse length color = 4 [
      set color (replace-item 3 color 0)
    ] [
      set color (lput 0 color)
    ]
  ] [
    set color extract-rgb color
    set color (lput 0 color)
  ]
end

to make-link-visible
  ifelse is-list? color [
    ifelse length color = 4 [
      set color (replace-item 3 color 255)
    ] [
      set color (lput 255 color)
    ]
  ] [
    set color extract-rgb color
    set color (lput 255 color)
  ]
end

to give-agent-ip-color [ agent ]
  ask agent [
    give-self-ip-color
  ]
end

to give-self-ip-color
  let a (dict-value brain "A")
  ifelse a = -1 [
    set color (extract-rgb gray)
  ] [
    let bel-color []
    set bel-color lput (255 - (round ((255 / (belief-resolution - 1)) * a))) bel-color
    set bel-color lput 0 bel-color
    set bel-color lput (round ((255 / (belief-resolution - 1)) * a)) bel-color
    set color bel-color
  ]
;  show (round (255 / belief-resolution) * a)


  ;; Attribute A color
;  if a = 0 [ set color (extract-rgb 12) ] ; dark red
;  if a = 1 [ set color (extract-rgb 14) ] ; red
;  if a = 2 [ set color (extract-rgb 16) ] ; light red
;  if a = 3 [ set color (extract-rgb 115) ] ; violet
;  if a = 4 [ set color (extract-rgb 106) ] ; light blue
;  if a = 5 [ set color (extract-rgb 104) ] ; blue
;  if a = 6 [ set color (extract-rgb 102) ] ; dark blue
end

to give-link-ip-color [ l ]
  ask l [
    give-self-link-ip-color
  ]
end

to give-self-link-ip-color
  let c1 [color] of end1
  let c2 [color] of end2
  let opacity 100
  ifelse c1 = c2 [
    ifelse length c1 = 4 [
      set color (replace-item 3 c1 opacity)
    ] [
      set color (lput opacity c1)
    ]
  ] [
    set color lput opacity (extract-rgb gray)
  ]
end

;; Lay out the simulation display based on the properties set in the simulation interface.
to layout
  update-agents

  ifelse show-media-connections? [
    ask subscribers [ make-link-visible ]
    ask media-peers [ make-link-visible ]
  ] [
    ask subscribers [ make-link-transparent ]
    ask media-peers [ make-link-transparent ]
  ]
  ifelse show-social-friends? [
    ask social-friends [
      make-link-visible
      give-self-link-ip-color
    ]
  ] [ ask social-friends [ make-link-transparent ] ]
end

;;;;;;;;;;;;;;;
; PROCS TO MATCH
; PY MESSAGING FILE
;;;;;;;;;;;;;;;

;; NOTE: For procedures that simply report back what comes from a python function, please refer
;; to the python function itself for function details.

;; Return a series of samples drawn from a normal distribution from [0, maxx]
;; with mean mu, std sigma; where each of en samples has k entries.
;; @param maxx - The maximum to draw from.
;; @param mu - The mean of the distribution.
;; @param sigma - The std deviation of the distribution.
;; @param en - The number of k entry samples to draw.
;; @param k - The number of entries per n sample.
to-report normal-dist-multiple [ maxx mu sigma en k ]
  report py:runresult(
    word "normal_dist_multiple(" maxx "," mu "," sigma "," en "," k ")"
  )
end

to-report sample-attr-dist [ attr ]
  report py:runresult(
    word "random_dist_sample(" attr "," belief-resolution ")"
  )
end

to-report sample-attr-dist-given [ attr given ]
  ;show(word "random_dist_sample(" attr "," (tuple-list-as-py-dict given false false) ")")
  ;; Now it's putting quotes around the Attribute.I which should not be there... have to reconcile somehow
  report py:runresult(
    word "random_dist_sample(" attr "," (tuple-list-as-py-dict given false false) ")"
  )
end

to-report create-agent-brain [ id prior-attrs malleable-attrs prior-vals malleable-vals ]
  report py:runresult(
    word "create_agent_brain(" id "," (list-as-py-array prior-attrs false) "," (list-as-py-array malleable-attrs false) "," (list-as-py-array prior-vals false) ", " (list-as-py-array malleable-vals false) ",'" brain-type "',1,1)"
  )
end

;; Change a belief in the agent brain structure.
;; @param agent-brain - The [brain] variable of the citizen agent type.
;; @param attr - The attribute to change.
;; @param value - The new value to update it to.
to-report update-agent-belief [ agent-brain attr value ]
  report py:runresult(
    (word "update_agent_belief(" (agent-brain-as-py-dict agent-brain) "," attr "," value ")")
  )
end

to-report random-message [ attrs ]
  report py:runresult(
    word "random_message(" (list-as-py-array attrs false) ")"
  )
end

to-report receive-message-py [ agent-brain message ]
  ;show(agent-brain-as-py-dict agent-brain)
  ;show(list-as-py-dict message false false)
  report py:runresult(
    word "receive_message(" (agent-brain-as-py-dict agent-brain) ", " (list-as-py-dict message true false) ", " spread-type ")"
  )
end

to-report believe-message-py [ agent-brain message ]
;  show(agent-brain-as-py-dict agent-brain)
  ;show(list-as-py-dict message false false)
;  show message
;  show (word "believe_message(" (agent-brain-as-py-dict agent-brain) ", " (list-as-py-dict message true false) ", '" spread-type "','" brain-type "')")
  report py:runresult(
    word "believe_message(" (agent-brain-as-py-dict agent-brain) ", " (list-as-py-dict message true false) ", '" spread-type "','" brain-type "')"
  )
end

to-report message-dist [ m1 m2 ]
  report py:runresult(
    word "message_distance(" (list-as-py-array m1 false) "," (list-as-py-array m2 false) ")"
  )
end

to-report weighted-message-dist [ m1 m2 m1-weights m2-weights ]
  report py:runresult(
    word "weighted_message_distance(" (list-as-py-array m1 false) "," (list-as-py-array m2 false) "," (list-as-py-array m1-weights false) "," (list-as-py-array m2-weights false) ")"
  )
end

to-report dist-to-agent-brain [ agent-brain message ]
  report py:runresult(
    word "dist_to_agent_brain(" (agent-brain-as-py-dict agent-brain) "," (list-as-py-dict message true false) ")"
  )
end

;; Get the distance between two agents' brains.
;; @param agent1-brain - The first agent's brain.
;; @param agent2-brain - The second agent's brain.
to-report dist-between-agent-brains [ agent1-brain agent2-brain ]
  report py:runresult(
    word "dist_between_agent_brains(" (agent-brain-as-py-dict agent1-brain) "," (agent-brain-as-py-dict agent2-brain) ")"
  )
end

to-report weighted-dist-to-agent-brain [ agent-brain message ]
  report py:runresult(
    word "weighted_dist_to_agent_brain(" (agent-brain-as-py-dict agent-brain) "," (list-as-py-dict message true false) "," cognitive-scalar ")"
  )
end

;; Create an Erdos-Renyi graph with the NetworkX package in python
;; @param en - The number of nodes for the graph (since N is a global variable)
;; @param p - The probability of two random nodes connecting.
;; @reports A dictionary [ ['nodes' nodes] ['edges' edges] ] where nodes is a list
;; of single values, and edges is a list of two-element lists (indicating nodes).
to-report er-graph [en p]
  report py:runresult((word "ER_graph(" en "," p ")"))
end

;; Create a Watts-Strogatz graph with the NetworkX package in python
;; @param en - The number of nodes for the graph (since N is a global variable)
;; @param k - The number of neighbors to initially connect to.
;; @param p - The probability of an edge rewiring.
;; @reports A dictionary [ ['nodes' nodes] ['edges' edges] ] where nodes is a list
;; of single values, and edges is a list of two-element lists (indicating nodes).
to-report ws-graph [en k p]
  report py:runresult((word "WS_graph(" en "," k "," p ")"))
end

;; Create a Barabasi-Albert graph with the NetworkX package in python
;; @param en - The number of nodes for the graph (since N is a global variable)
;; @param m - The number of edges to connect with when a node is added.
;; @reports A dictionary [ ['nodes' nodes] ['edges' edges] ] where nodes is a list
;; of single values, and edges is a list of two-element lists (indicating nodes).
to-report ba-graph [en m]
  report py:runresult((word "BA_graph(" en "," m ")"))
end

;; Run a MAG function in the python script.
;; @param en - The number of nodes for the graph (since N is a global variable)
;; @param attrs - A list of attributes to construct the graph from - these will designate
;; attribute affinity matrices defined in the data.py file to use for edge probabilities.
;; @param style - A connection style to use if no more specific setup is designated.
;; @param belief-resolution - A parameterized setting to denote how finely to break up belief scales.
to-report mag [ en attrs style ]
  report py:runresult(
    (word "MAG_graph(" en "," attrs ",'" style "'," belief-resolution ")")
  )
end

to-report kronecker [ seed k ]
  report py:runresult(
    (word "kronecker_graph_bidirected(np.array(" seed ")," k ")")
  )
end

;; Connect a MAG graph based on values in the global mag-g variable (those values
;; are probabilities that two nodes in a graph will connect).
to connect_mag
  let u 0
  let v 0
  foreach mag-g [ row ->
     set v 0
     foreach row [ el ->
      let rand random-float 1
      if (el > rand) and (u != v) [
        ;show(word "Linking turtle b/c el:" el " and rand " rand)
        ask turtle u [ create-social-friend-to turtle v [ set weight citizen-citizen-influence ] ]
      ]
      set v v + 1
    ]
    set u u + 1
  ]
end

;; Create a graph of agents and links with the help of the NetworkX python package. This procedure
;; is not parameterized, but rather takes the citizens and social edges in the existing network.
;;
;; @reports A NetworkX graph object (which ends up being a NetLogo dictionary of nodes and edges).
to-report nx-graph
  let citizen-arr list-as-py-array (map [ cit -> agent-brain-as-py-dict [brain] of citizen cit ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  report py:runresult(
    (word "nlogo_graph_to_nx(" citizen-arr "," edge-arr ")")
  )
end

;; Calculate paths between influencers and target agents given a message and a threshold. This will return paths
;; that only contain agents of threshold distance from the message.
;;
;; @param influencer - The influencer agent (w_0 of the path)
;; @param target - The target agent (w_k of the path)
;; @param message - The message being compared to each agent in the paths
;; @param t - The threshold of distance between agent brain and message to use for path counts
;; @reports A list of paths only containing agents within threshold distance of the message.
to-report influencer-distance-paths [ influencer target message t ]
  let citizen-arr list-as-py-array (map [ cit -> agent-brain-as-py-dict [brain] of citizen cit ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  let subscribers-arr list-as-py-array (sort [subscribers] of influencer) true
  let message-dict list-as-py-dict message true false
;  report (word "influencer_paths_within_distance(" citizen-arr "," edge-arr "," subscribers-arr ",'" target "'," message-dict "," t ")")
  report py:runresult(
    (word "influencer_paths_within_distance(" citizen-arr "," edge-arr "," subscribers-arr ",'" target "'," message-dict "," t ")")
  )
end

;; Find a community in the graph that can become the Flint community based on
;; community detection with the Louvain algorithm (ideally of size n).
;;
;; @param n -- The number of citizens ideally to look for in a community
to-report flint-community [ comm en ]
  report py:runresult(
    (word "flint_community(" (list-of-dicts-as-py-list comm false false) "," en")")
  )
end

to-report communities-by-level
  let citizen-arr list-as-py-array (map [ cit -> agent-brain-as-py-dict [brain] of citizen cit ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  report py:runresult(
    (word "nlogo_communities_by_level(" citizen-arr "," edge-arr ")")
  )
end

to-report community-sizes-by-level [ comm ]
  report map [ level -> max (map [ cit-comm-pair -> item 1 cit-comm-pair ] level) ] comm
end

;;
;; @param cit -- The citizen id to get degree centrality for.
to-report citizen-degree-centrality [ cit-id ]
  let citizen-arr list-as-py-array (map [ c -> agent-brain-as-py-dict [brain] of citizen c ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  report py:runresult(
    (word "node_degree_centrality(nlogo_graph_to_nx(" citizen-arr "," edge-arr ")," cit-id ")")
  )
end

;;
;; @param cit -- The citizen id to get degree centrality for.
to-report citizens-degree-centrality [ cit-ids ]
  let citizen-arr list-as-py-array (map [ c -> agent-brain-as-py-dict [brain] of citizen c ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  report py:runresult(
    (word "nodes_degree_centrality(nlogo_graph_to_nx(" citizen-arr "," edge-arr ")," (list-as-py-array cit-ids false) ")")
  )
end

to-report media-peer-connections
  let citizen-arr list-as-py-array (map [ cit -> agent-brain-as-py-dict [brain] of citizen cit ] (range N)) false
  let edge-arr list-as-py-array (sort social-friends) true
  let media-arr list-as-py-array (map [ med -> agent-brain-as-py-dict [brain] of med ] (sort medias)) false
  let subscriber-arr list-as-py-array (sort subscribers) true
;  show (word "nlogo_graph_to_nx_with_media(" citizen-arr "," edge-arr "," media-arr "," subscriber-arr ")")
  report py:runresult(
    (word "media_peer_connections(nlogo_graph_to_nx_with_media(" citizen-arr "," edge-arr "," media-arr "," subscriber-arr "))")
  )
end

;; Write out the adoption-related data that the simulation has stored
to output-adoption-data [ path uniqueid ]
  let messages-adopted-py-str (list-as-py-array (map [ tick-entry -> (word (item 0 tick-entry) ": " list-as-py-dict (item 1 tick-entry) true true) ] agents-adopted-by-tick) false)
  let messages-adopted-py (word "{" (substring messages-adopted-py-str 1 ((length messages-adopted-py-str) - 1)) "}" )
  py:run (word "write_message_data('" path "'," "'" uniqueid "'," messages-adopted-py ")")
end

;;;;;;;;;;;;;;;
; HELPER PROCS
;;;;;;;;;;;;;;;

to-report is-agent-brain-empty? [ agent ]
  report empty? agent-brain-malleable-values agent
end

to-report array_shape [g]
  report py:runresult(
    word "kron.np.array(" g ").shape[0]"
  )
end

;; Get the value of an attribute from the Attributes enumeration in Python.
;;
;; @param attr - The variable name of the attribute.
;; @param val - The value of the attribute to fetch.
;; @reports The integer value associated with the enumeration.
to-report name-of-attribute-val [ attr val ]
  report py:runresult(
    word "Attributes." attr ".value(" val ").name"
  )
end

to-report agent-brain-as-py-dict [ b ]
  ;; Convert to a py-dict
  report list-as-py-dict-rec b true false
end

to-report agent-brain-malleable-values [ agent ]
  let b [brain] of agent
  let malleables (dict-value b "malleable")
  report filter [ bel -> member? (item 0 bel) malleables ] [brain] of agent
end

;; Limits a value between a min and a max.
;; @param val - The value.
;; @param lower - The lower bound.
;; @param upper - The upper bound.
;; @reports The value squeezed between the two bounds.
to-report squeeze [ val lower upper ]
  report (max (list (min list val upper) lower))
end

;; @reports a date/time string that is safe for file names
to-report date-time-safe
  let datetime date-and-time
  let safedatetime ""
  let i 0
  repeat length datetime [
    let char item i datetime
    if char != " " and char != ":" and char != "." [
      set safedatetime (word safedatetime char)
    ]
    set i i + 1
  ]
  report safedatetime
end

;;;;;;;;;;;;;;;;;;
;; PROCS TO HELP
;; WITH PYTHON CONVERSION
;;;;;;;;;;;;;;;;;;

to-report replace-dict-item [ l key value ]
  let key-i 0
  let i 0
  foreach l [ el ->
    if (item 0 el) = key [
      set key-i i
    ]
    set i i + 1
  ]
  report (replace-item key-i l value)
end

to-report dict-value [ dict key ]
  foreach dict [ list-attr ->
    if item 0 list-attr = key[
      report item 1 list-attr
    ]
  ]
  report -1
end

to-report dict-entry [ dict key ]
  foreach dict [ list-attr ->
    if item 0 list-attr = key [
      report list-attr
    ]
  ]
  report -1
end

to-report index-of-dict-entry [ dict key ]
  let i 0
  foreach dict [ list-attr ->
    if item 0 list-attr = key [
      report i
    ]
    set i i + 1
  ]
  report -1
end

to-report list-as-py-array [ l val-quotes? ]
  let py-array "["
  let i 1
  foreach l [ el ->
    if val-quotes? [ set el (word "'" el "'") ]

    ifelse i = length l
    [ set py-array (word "" py-array el) ]
    [ set py-array (word "" py-array el ",") ]

    set i i + 1
  ]
  report (word py-array "]")
end

to-report list-item-as-dict-item [ el key-quotes? val-quotes? ]
  if key-quotes? and val-quotes? [ report (word "'" (item 0 el) "': '" (item 1 el) "'") ]
  if key-quotes? [ report (word "'" (item 0 el) "': " (item 1 el)) ]
  if val-quotes? [ report (word (item 0 el) ": '" (item 1 el) "'") ]
  report (word (item 0 el) ": " (item 1 el))
end

to-report multi-list-as-py-dict [ ml key-quotes? val-quotes? ]
  let attr (item 0 ml)
  let l (item 1 ml)
  let py-dict (word  "{'" attr "': { ")
  let i 1
  foreach l [ el ->
    ;show(tuple-list-as-py-dict el)
    ifelse i = length l
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) " }") ]
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) ", ") ]

    set i i + 1
    ;show(py-dict)
  ]
  report py-dict
end

to-report multi-list-as-tuple-list [ ml key-quotes? val-quotes? ]
  let attr (item 0 ml)
  let l (item 1 ml)
  let py-dict "{ "
  let i 1
  foreach l [ el ->
    ;show(tuple-list-as-py-dict el)
    ifelse i = length l
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) " }") ]
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) ", ") ]

    set i i + 1
    ;show(py-dict)
  ]
  report list attr py-dict
end

;; In the case that there is a list of dictionaries in nlogo (e.g. [ ['a': 1, 'b': 2], ['c': 3] ]),
;; convert it into a python syntax list of dictionaries
to-report list-of-dicts-as-py-list [ l key-quotes? val-quotes? ]
  let py-list "[ "
  let i 1
  foreach l [ dict ->
    ifelse i = length l
    [ set py-list (word py-list (list-as-py-dict dict key-quotes? val-quotes?) " ]") ]
    [ set py-list (word py-list (list-as-py-dict dict key-quotes? val-quotes?) ",") ]

    set i i + 1
  ]
  report py-list
end

to-report list-as-py-dict [ l key-quotes? val-quotes? ]
  let py-dict "{ "
  let i 1

  if empty? l [ set py-dict (word py-dict "}") ]

  foreach l [ el ->
    ;show(tuple-list-as-py-dict el)
    ifelse i = length l
    ;[ set py-dict (word py-dict (tuple-list-as-py-dict el) " }") ]
    ;[ set py-dict (word py-dict (tuple-list-as-py-dict el) ", ") ]
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) " }") ]
    [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) ", ") ]

    set i i + 1
    ;show(py-dict)
  ]
  report py-dict
end

; [ [ 'key1' val ] [ 'key2' [ 'key3' val2 ] ] [ 'key4' [ val3 val4 val5 ] ] ]
to-report list-as-py-dict-rec [ l key-quotes? val-quotes? ]
  let py-dict "{ "
  let i 1
  foreach l [ el ->
    ifelse length el = 2 and is-list? item 1 el and length item 1 el > 0 [
      ifelse is-list? (item 0 (item 1 el)) [
        let temp-item list (item 0 el) (list-as-py-dict-rec (item 1 el) key-quotes? val-quotes?)
        ifelse i = length l
        [ set py-dict (word py-dict (list-item-as-dict-item temp-item key-quotes? val-quotes?) " }") ]
        [ set py-dict (word py-dict (list-item-as-dict-item temp-item key-quotes? val-quotes?) ", ") ]
      ] [
        let temp-item list (item 0 el) (list-as-py-array item 1 el true)
        ifelse i = length l
        [ set py-dict (word py-dict (list-item-as-dict-item temp-item key-quotes? false) " }") ]
        [ set py-dict (word py-dict (list-item-as-dict-item temp-item key-quotes? false) ", ") ]
      ]
    ] [
      ifelse i = length l
      [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) " }") ]
      [ set py-dict (word py-dict (list-item-as-dict-item el key-quotes? val-quotes?) ", ") ]
    ]

    set i i + 1
    ;show(py-dict)
  ]
  report py-dict
end

to-report tuple-list-as-py-dict [ l key-quotes? val-quotes? ]
  if (length l = 2) [
    report (word "{"(list-item-as-dict-item l key-quotes? val-quotes?) "}")
  ]
  report -1
end
@#$#@#$#@
GRAPHICS-WINDOW
1060
14
1689
644
-1
-1
18.82
1
10
1
1
1
0
1
1
1
-16
16
-16
16
0
0
1
ticks
30.0

BUTTON
21
55
84
88
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
172
56
279
90
Reset Python
setup-py
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
21
98
84
131
Step
step
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
1063
718
1460
911
A Histogram
A Value
Number of Agents
-4.0
4.0
0.0
50.0
true
false
"" ""
PENS
"default" 1.0 1 -16777216 true "" "plot-pen-reset  ;; erase what we plotted before\nset-plot-x-range -1 (belief-resolution + 1)\n\nhistogram [dict-value brain \"A\"] of citizens"

MONITOR
1059
919
1117
964
0
count citizens with [dict-value brain \"A\" = 0]
1
1
11

MONITOR
1118
919
1175
964
1
count citizens with [dict-value brain \"A\" = 1]
1
1
11

MONITOR
1179
919
1245
964
2
count citizens with [dict-value brain \"A\" = 2]
1
1
11

MONITOR
1250
919
1308
964
3
count citizens with [dict-value brain \"A\" = 3]
1
1
11

MONITOR
1308
919
1366
964
4
count citizens with [dict-value brain \"A\" = 4]
1
1
11

BUTTON
95
55
158
88
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

SLIDER
528
722
658
755
threshold
threshold
0
20
20.0
1
1
NIL
HORIZONTAL

SLIDER
178
314
350
347
epsilon
epsilon
0
100
0.0
0.1
1
NIL
HORIZONTAL

SWITCH
298
58
495
91
show-media-connections?
show-media-connections?
1
1
-1000

BUTTON
95
98
161
131
NIL
layout
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

TEXTBOX
29
432
179
450
Number of citizens
11
0.0
1

TEXTBOX
178
294
344
322
Threshold to subscribe to media
11
0.0
1

TEXTBOX
25
144
175
162
Simulation Parameters
14
0.0
1

TEXTBOX
23
17
173
35
Simulation Controls
14
0.0
1

TEXTBOX
1064
660
1214
678
Simulation State Plots
14
0.0
1

SLIDER
247
460
419
493
N
N
0
10000
300.0
10
1
NIL
HORIZONTAL

SWITCH
502
58
676
91
show-citizen-political?
show-citizen-political?
0
1
-1000

SWITCH
300
99
467
132
show-social-friends?
show-social-friends?
0
1
-1000

TEXTBOX
1063
688
1213
706
Cognitive State
11
0.0
1

PLOT
1069
1014
1338
1249
Social Friend Degree of Nodes
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 1 -16777216 true "" "set-plot-x-range 0 (max [count social-friend-neighbors] of citizens) + 1\nhistogram [count social-friend-neighbors] of citizens"

TEXTBOX
1065
983
1253
1006
Aggregate Charts
13
0.0
1

CHOOSER
330
775
472
820
spread-type
spread-type
"simple" "complex" "cognitive"
0

TEXTBOX
302
35
490
58
Display
11
0.0
1

SWITCH
29
455
148
488
load-graph?
load-graph?
1
1
-1000

INPUTBOX
27
495
242
555
load-graph-path
D:/school/grad-school/Tufts/research/flint-media-model/simulation-data/18-Apr-2023-static-no-organizing-media-connect-sweep/graphs/0.5-3-0.75-0.75-4.csv
1
0
String

INPUTBOX
29
562
244
622
save-graph-path
D:/school/grad-school/Tufts/research/flint-media-model/simulation-data/18-Apr-2023-static-no-organizing-media-connect-sweep/graphs/0.5-3-0.75-0.25-4.csv
1
0
String

BUTTON
173
97
270
131
Save Graph
save-graph
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

CHOOSER
20
774
173
819
cognitive-fn
cognitive-fn
"linear-gullible" "linear-stubborn" "linear-mid" "threshold-gullible" "threshold-mid" "threshold-stubborn" "sigmoid-gullible" "sigmoid-stubborn" "sigmoid-mid"
7

SLIDER
165
720
339
753
simple-spread-chance
simple-spread-chance
0
1
0.51
0.01
1
NIL
HORIZONTAL

SLIDER
348
722
522
755
complex-spread-ratio
complex-spread-ratio
0
1
0.01
0.01
1
NIL
HORIZONTAL

CHOOSER
185
775
324
820
brain-type
brain-type
"discrete" "continuous"
0

SLIDER
25
175
199
208
tick-end
tick-end
30
1000
114.0
1
1
NIL
HORIZONTAL

INPUTBOX
24
215
341
277
sim-output-dir
D:/school/grad-school/Tufts/research/flint-media-model/simulation-data/
1
0
String

PLOT
1702
280
2076
568
percent-agent-beliefs
Steps
% of Agents
0.0
10.0
0.0
1.0
true
false
"let i 0\nrepeat belief-resolution [\n  let pen-name (word i)\n  create-temporary-plot-pen pen-name\n  set-current-plot-pen pen-name\n  \n  let bel-color []\n  set bel-color lput (255 - (round ((255 / (belief-resolution - 1)) * i))) bel-color\n  set bel-color lput 0 bel-color\n  set bel-color lput (round ((255 / (belief-resolution - 1)) * i)) bel-color\n\n  set-plot-pen-color bel-color\n\n  set i i + 1\n]" "let i 0\nrepeat belief-resolution [\n  let pen-name (word i)\n  set-current-plot-pen pen-name\n  \n  plot (count citizens with [ dict-value brain \"A\" = i ]) / (count citizens)\n\n  set i i + 1\n]"
PENS

MONITOR
1363
919
1421
964
5
count citizens with [dict-value brain \"A\" = 5]
17
1
11

MONITOR
1424
919
1482
964
6
count citizens with [dict-value brain \"A\" = 6]
17
1
11

SWITCH
28
310
162
343
media-agents?
media-agents?
0
1
-1000

SLIDER
20
864
193
897
cognitive-exponent
cognitive-exponent
-10
10
4.0
1
1
NIL
HORIZONTAL

SLIDER
20
824
193
857
cognitive-scalar
cognitive-scalar
-20
20
20.0
1
1
NIL
HORIZONTAL

SWITCH
200
824
345
857
cognitive-scalar?
cognitive-scalar?
1
1
-1000

SWITCH
204
865
369
898
cognitive-exponent?
cognitive-exponent?
0
1
-1000

SLIDER
20
909
193
942
cognitive-translate
cognitive-translate
-10
20
2.0
1
1
NIL
HORIZONTAL

SWITCH
204
909
367
942
cognitive-translate?
cognitive-translate?
0
1
-1000

TEXTBOX
24
697
212
720
Contagion Parameters
11
0.0
1

CHOOSER
247
499
386
544
graph-type
graph-type
"erdos-renyi" "watts-strogatz" "barabasi-albert" "mag" "facebook" "kronecker"
2

SLIDER
255
573
378
606
erdos-renyi-p
erdos-renyi-p
0
1
0.05
0.01
1
NIL
HORIZONTAL

TEXTBOX
28
285
216
308
Influencer Parameters
11
0.0
1

TEXTBOX
30
410
218
433
Graph Parameters
11
0.0
1

SLIDER
383
573
517
606
watts-strogatz-p
watts-strogatz-p
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
383
613
510
646
watts-strogatz-k
watts-strogatz-k
0
N - 1
10.0
1
1
NIL
HORIZONTAL

SLIDER
254
612
379
645
ba-m
ba-m
0
50
3.0
1
1
NIL
HORIZONTAL

CHOOSER
524
574
663
619
mag-style
mag-style
"default" "homophilic" "heterophilic"
0

SWITCH
25
720
158
753
contagion-on?
contagion-on?
0
1
-1000

SLIDER
497
99
670
132
belief-resolution
belief-resolution
0
100
7.0
1
1
NIL
HORIZONTAL

SLIDER
360
359
538
392
citizen-citizen-influence
citizen-citizen-influence
0
1
0.75
0.01
1
NIL
HORIZONTAL

SLIDER
360
315
535
348
citizen-media-influence
citizen-media-influence
0
1
0.75
0.01
1
NIL
HORIZONTAL

SLIDER
360
400
535
433
media-citizen-influence
media-citizen-influence
0
1
1.0
0.01
1
NIL
HORIZONTAL

TEXTBOX
363
295
513
313
Link weight settings
11
0.0
1

SLIDER
860
465
1020
498
flint-community-size
flint-community-size
0
1
0.005
0.001
1
NIL
HORIZONTAL

PLOT
1702
15
2079
274
num-new-beliefs
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot num-agents-adopted"

SWITCH
713
315
928
348
dynamic-cit-media-influence?
dynamic-cit-media-influence?
1
1
-1000

SWITCH
544
317
701
350
cit-media-gradual?
cit-media-gradual?
1
1
-1000

SLIDER
545
357
705
390
cit-media-gradual-scalar
cit-media-gradual-scalar
1
10
1.0
1
1
NIL
HORIZONTAL

PLOT
1702
579
2062
819
degree-centrality-flint
NIL
NIL
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot citizens-degree-centrality [ (dict-value brain \"ID\") ] of citizens with [is-flint?]"

SWITCH
874
505
1014
538
flint-organizing?
flint-organizing?
1
1
-1000

CHOOSER
675
499
869
544
flint-organizing-strategy
flint-organizing-strategy
"high-degree-media" "high-degree-citizens" "neighbors-of-neighbors" "high-degree-cit-and-media"
3

SLIDER
29
355
207
388
media-connection-prob
media-connection-prob
0
1
0.25
0.01
1
NIL
HORIZONTAL

MONITOR
1480
719
1596
764
Number of Media
count medias
17
1
11

SLIDER
677
460
850
493
organizing-capacity
organizing-capacity
0
50
10.0
1
1
NIL
HORIZONTAL

TEXTBOX
255
553
443
576
Graph specific parameters
11
0.0
1

SWITCH
713
358
906
391
dynamic-cit-cit-influence?
dynamic-cit-cit-influence?
1
1
-1000

TEXTBOX
680
439
868
462
Citizen behaviors
11
0.0
1

SLIDER
455
447
629
480
media-media-influence
media-media-influence
0
1
0.5
0.01
1
NIL
HORIZONTAL

SWITCH
589
214
762
247
media-monitor-peers?
media-monitor-peers?
1
1
-1000

SLIDER
209
175
382
208
repetition
repetition
0
50
4.0
1
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="static-influence-sweep" repetitions="30" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "static-influence-sweep")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")
output-adoption-data contagion-dir behavior-rand</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="3"/>
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
    <steppedValueSet variable="repetition" first="0" step="1" last="4"/>
  </experiment>
  <experiment name="static-organizing-sweep" repetitions="30" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "static-organizing-sweep")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="3"/>
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cit-media-gradual?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="organizing-capacity">
      <value value="1"/>
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing-strategy">
      <value value="&quot;neighbors-of-neighbors&quot;"/>
      <value value="&quot;high-degree-media&quot;"/>
      <value value="&quot;high-degree-citizens&quot;"/>
      <value value="&quot;high-degree-cit-and-media&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="base-model-sweep" repetitions="30" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "base-model-sweep")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="3"/>
      <value value="10"/>
      <value value="20"/>
      <value value="50"/>
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
    <steppedValueSet variable="repetition" first="0" step="1" last="4"/>
  </experiment>
  <experiment name="static-no-organizing-media-connect-sweep" repetitions="30" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "static-no-organizing-media-connect-sweep")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="3"/>
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cit-media-gradual?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
    <steppedValueSet variable="repetition" first="0" step="1" last="4"/>
  </experiment>
  <experiment name="static-organizing-media-connect-sweep" repetitions="10" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "static-organizing-media-connect-sweep")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="3"/>
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cit-media-gradual?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="0.01"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="organizing-capacity">
      <value value="1"/>
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing-strategy">
      <value value="&quot;neighbors-of-neighbors&quot;"/>
      <value value="&quot;high-degree-media&quot;"/>
      <value value="&quot;high-degree-citizens&quot;"/>
      <value value="&quot;high-degree-cit-and-media&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
    <steppedValueSet variable="repetition" first="0" step="1" last="2"/>
  </experiment>
  <experiment name="static-influence-monte-carlo-1" repetitions="10" runMetricsEveryStep="false">
    <setup>setup-py
let run-dir (word sim-output-dir "static-influence-monte-carlo-1")
let graphs-path (word run-dir "/graphs")
carefully [
  if not (py:runresult (word "os.path.isdir('" graphs-path "')")) [
    py:run (word "create_nested_dirs('" graphs-path "')")
  ]
] [ ]
let graph-file (word graphs-path "/" simple-spread-chance "-" ba-m "-" citizen-media-influence "-" citizen-citizen-influence "-" repetition ".csv")
ifelse (py:runresult (word "os.path.isfile('" graph-file "')")) [
  set load-graph? true
  set load-graph-path graph-file
  setup
] [
  set load-graph? false
  set save-graph-path graph-file
  setup
  save-graph
]
set contagion-dir (word run-dir "/" simple-spread-chance "/" ba-m "/" citizen-media-influence "/" citizen-citizen-influence "/" repetition)
carefully [
  if not (py:runresult (word "os.path.isdir('" contagion-dir "')")) [
    py:run (word "create_nested_dirs('" contagion-dir "')")
  ]
] [ ]</setup>
    <go>go</go>
    <final>set behavior-rand random 10000
export-world (word contagion-dir "/" behavior-rand "_world.csv")
export-plot "percent-agent-beliefs" (word contagion-dir "/" behavior-rand "_percent-agent-beliefs.csv")
export-plot "num-new-beliefs" (word contagion-dir "/" behavior-rand "_new-beliefs.csv")
output-adoption-data contagion-dir behavior-rand</final>
    <timeLimit steps="114"/>
    <metric>count citizens</metric>
    <enumeratedValueSet variable="contagion-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-cit-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-cit-media-influence?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-monitor-peers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-organizing?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="load-graph?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="media-agents?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="belief-resolution">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="brain-type">
      <value value="&quot;discrete&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="N">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tick-end">
      <value value="114"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="spread-type">
      <value value="&quot;simple&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simple-spread-chance">
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="graph-type">
      <value value="&quot;barabasi-albert&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ba-m">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="epsilon">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-media-influence">
      <value value="0.01"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="citizen-citizen-influence">
      <value value="0.75"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="flint-community-size">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="repetition">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
