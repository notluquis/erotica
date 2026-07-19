# Referee report — round 2 (received 2026-07-16)

Report by referee 1 on the revised version of aa52082-24.

## General

> Following the editor's request, I read the revised version of the manuscript by
> Pulgar-Escobar et al. Unfortunately, the paper is extremely hard to read and lacks
> clarity in many aspects. These problems are due to a large part to the overall
> structure adopted by the authors. Section 2 gives a longish description of a suite
> of (mostly) preexisting tools that were used, but the logical flow of the overall
> methodology is unclear or even lacking. Adding a paragraph explaining FOR WHAT
> PURPOSE some of the tools are used and why their use is a logical consequence of
> the previous steps of the analysis would definitely help. Figures with results
> appear as early as page 2, but are only vaguely described starting on page 7! In my
> opinion, it would be much better to re-organise the paper by presenting the results
> of the application of the various tools to NGC6383 directly after the description
> of the tools. Furthermore, the tools used by the authors rely on numerical
> parameters that are rarely justified. The authors should state how these parameters
> were chosen and what are the implications of those choices?

## Specific comments (numbered for tracking)

R1. **Sect. 2.1 tool enumeration** — COSMIC, NUTS, DEMetropolis etc.: unclear what
each is used FOR; sentences not a scientific description; add website/reference for
each tool.

R2. **Sect. 2.1.1 native HDBSCAN membership probability** — probabilities are the
product of recovery frequencies and a "native HDBSCAN membership probability" that
is never explained. How obtained?

R3. **Sect. 2.1.1 lambda** — density level parameter λ introduced as diagnostic of
branch persistence/separation, but totally unclear how it is used (or not) in the
analysis.

R4. **Sect. 2.1.2 notation** — \mathcal{U} used (per response letter = half-normal):
non-standard, must be CLEARLY introduced at first use. Also \bar{\varpi} argument
never specified.

R5. **Table 1 "Distance (D.M.)"** — meaning of the 2nd line? Inferred from
Bailer-Jones distances? Clarify.

R6. **Sect. 2.1.3 sampler diagnostics** — values like energy-BFMI never introduced;
opaque to a general reader.

R7. **Sect. 2.1.5 prior choices** — 0.8 Tmax, 1.5 Tmax etc. appear arbitrary. Why
not 0.9/1.2? Impact on outcome? Same for alpha and beta parameters in Sect. 2.2
(binarity): choice unexplained; does it match observed binarity vs spectral
type/mass?

R8. **Sect. 2.1.5 galactocentric distance 7.19 kpc** — how obtained?

R9. **Sect. 2.2 MIST assumptions** — ages extremely sensitive to evolutionary-code
assumptions (overshooting etc.), especially very young clusters; recall the most
important assumptions behind MIST.

R10. **Sect. 2.2 last sentence** — "nuisance parameter shown in the posterior
diagnostic plot": which figure number? (B.4?) Specify.

R11. **Sect. 3 extraction radius vs King radius** — text claims structural analyses
use fields substantially larger than cluster radius, but 40′ extraction < R_t
40.4±14.3′. Instead of paragraphs justifying 40′, why not USE data from a larger
extraction radius? Conclusion itself admits outer membership not invariant with
field size.

R12. **Sect. 3.4 2nd paragraph** — "only the first comparison survives... than a
dynamical one" incomprehensible; reformulate.

R13. **Sect. 3.4 secondary masses** — "while secondary masses are added to the
system mass...": how were they added? What binary mass ratio assumed?

R14. **Sect. 3.4 t_seg = 2.94 Myr** — depends strongly on most massive star; would
differ if HD 159176 were a member — point this out. Also: "evaluate the potential
for mass segregation by comparing the radial distribution with a segregation time
greater than the 3.53 Myr cluster age" + "0.19–8.49 Msun = 99.2% of the reference
sample" — incomprehensible given t_seg < age; clarify what is actually done.

R15. **Sect. 3.5.1 "accreting Be-binary scenario"** — ambiguous; HD 159176 is not a
Be HMXB nor a mass-transfer binary with a Be gainer. Back up with reference or fix.

R16. **Sect. 3.5.3 organization** — finish Kalari (2019) discussion BEFORE moving to
Rauw et al. (2010); also say HOW compared literature ages were obtained
(evolutionary-code assumptions → age differences).

R17. **Sect. 3.5.4 added value** — vs published catalogs: in what respect is this
study superior to past work? Especially given code-parameter choices that might
impact member selection / cluster parameters.
