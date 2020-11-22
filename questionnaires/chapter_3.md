## Questionnaire: Data Ethics

---
1. Does ethics provide a list of "right answers"?
---
No, ethics provide a framework to define ethical standards, but these are dynamic and not are universal true.

---
2. How can working with people of different backgrounds help when considering ethical questions?
---
As what is right or wrong is not universal, having different points of view help to identify ethical problems.

---
3. What was the role of IBM in Nazi Germany? Why did the company participate as it did? Why did the workers participate?
---
IBM provided the machines to manage the logistics and keep records of prisoners in concentration camps, facilitating the crimes against humanity. The company did it for profit and workers simply followed orders from their superiors. 

---
4. What was the role of the first person jailed in the Volkswagen diesel scandal?
---
It was one of the engineers. This is an example of how technologists are accountable for what we do and must question how our work affects others.

---
5. What was the problem with a database of suspected gang members maintained by California law enforcement officials?
---
It had people in the database that couldn't be gang members as babies and no process in place to remove registers or fix errors.

---
6. Why did YouTube's recommendation algorithm recommend videos of partially clothed children to pedophiles, even though no employee at Google had programmed this feature?
---
The algorithm recommends videos with similar features to the user's history creating a feedback loop. If a pedophile watches videos with partially clothed children, the recommender will suggest similar ones to him and other similar users and more pedophiles will be watching these kind of videos, these videos will have more views and therefore more visibility. For more details, see this recent [post](https://www.huffpost.com/entry/youtube-pedophile-paradise_n_5e5d79d1c5b6732f50e6b4db).

---
7. What are the problems with the centrality of metrics?
---
That algorithms try hard to optimize them and that can cause unexpected consequences against our ethical values.

---
8. Why did Meetup.com not include gender in its recommendation system for tech meetups?
---
To not reinforce the gender gap in tech.

---
9. What are the six types of bias in machine learning, according to Suresh and Guttag?
---
- Historical bias: mismatch between old and current values
- Representation bias: mismatch between the population definition and the sample
- Measurement bias: mismatch between the variables and the features used to measure them
- Aggregation bias: mismatch between combined subgroups
- Evaluation bias: mismatch between the model goals and the metrics established or the population used to measure them
- Deployment bias: mismatch between the initial model purpose and the way it's used

Defined in the paper [A Framework for Understanding Unintended Consequences of Machine Learning](https://arxiv.org/pdf/1901.10002.pdf)

---
1.  Give two examples of historical race bias in the US.
---
- Black people get higher prices when bargaining for a used car
- An all-white jury is more likely to sentence a black person than a jury with one black member

---
11. Where are most images in ImageNet from?
---
Most images are from US and Europe

---
12. In the paper "Does Machine Learning Automate Moral Hazard and Error" why is sinusitis found to be predictive of a stroke?
---
The model is not predicting who has a stroke but who has the symptoms and went to the doctor, which can be a person with sinusitis.

---
13. What is representation bias?
---
This bias arises when subgroups of the population are underrepresented. In this case, the minority class will be predicted with a lower proportion that in the real population. 

---
14. How are machines and people different, in terms of their use for making decisions?
---
Algorithms are more easy to trust than humans and we can't complain or appeal their decisions. Apart from that, they are cheaper and therefore easy to scale, so one bad algorithm can have a high impact on the society.

---
15. Is disinformation the same as "fake news"?
---
No, disinformation may contains some truth, half lies, exaggerations as its intention is to create uncertainty.

---
16. Why is disinformation through auto-generated text a particularly significant issue?
---
Because recent language models powered by deep learning can generate text that look like written by a person, which makes easier to scale disinformation campaigns and fake news.

---
17. What are the five ethical lenses described by the Markkula Center?
---
Lenses to identify potential ethical issues and their consequences.

- The rights approach: how to best ensure respect for human rights 
- The justice approach: how to treat people equally
- The utilitarian approach: how to benefit people
- The common good approach: how to benefit most people
- The virtue approach: how to do something we can be proud of

---
18.  Where is policy an appropriate tool for addressing data ethics issues?
---
Where coordinations among different organizations, states or countries is required. Companies are not `motivated` to act unless there is an economical or legal penalty.

## Further Research:

---
1. Read the article "What Happens When an Algorithm Cuts Your Healthcare". How could problems like this be avoided in the future?
---
- Fairness: Before implementing an algorithm, it should be checked that people with the same condition get equal treatment (hours per week in this case).
- Accountability: Before implementation, values given by the algorithm and the current values should be compared and investigate reasons for major discrepancies. The algorithm should be able to explain a change in the hours a person is getting. 
- Transparency: the formula should be public, so that it can be reviewed and challenged.

---
2. Research to find out more about YouTube's recommendation system and its societal impacts. Do you think recommendation systems must always have feedback loops with negative results? What approaches could Google take to avoid them? What about the government?
---
No, randomness should be a part of any recommendation system to reduce exponential behavior and expose users to a more real life environment where we can normally find different ideas or content. Also, there should be some restrictions for the recommendations, in this case for example, some videos shouldn't be recommended to users with suspicious activity and they should get recommendations at random and suspicious videos shouldn't be recommended more than a defined limit of times.

---
3. Read the paper "Discrimination in Online Ad Delivery". Do you think Google should be considered responsible for what happened to Dr. Sweeney? What would be an appropriate response?
---
Google should be considered responsible as they didn't make sure their algorithms were fair and transparent. An appropriate response would be to conduct an internal study to explain why common names for black people should trigger more often the word `arrest` and take measures to control this behavior.

---
4. How can a cross-disciplinary team help avoid negative consequences?
---
A cross-disciplinary team can identify possible negative consequences as every member of the team may have experienced diverse situations and can evaluate every option seeing from a different lense.

---
5. Read the paper "Does Machine Learning Automate Moral Hazard and Error". What actions do you think should be taken to deal with the issues identified in this paper?
---
- To reduce measurement bias data collected for a specific purpose should be collected. Medical registers contains subjective and selective information, therefore, they can't be used always to traing ML models.
- To reduce evaluation bias sampling or randomized trials can be used for validation and predictions are compared to standard measures.

---
6. Read the article "How Will We Prevent AI-Based Forgery?" Do you think Etzioni's proposed approach could work? Why?
---
It could work if a decentralized and non-profit organization is controlling the `signatures` or else this could lead to censorship and few people with obscure interests defining which information is `authentic`.

---
7. Complete the section "Analyze a Project You Are Working On" in this chapter.
---
NA

---
8. Consider whether your team could be more diverse. If so, what approaches might help?
---
NA