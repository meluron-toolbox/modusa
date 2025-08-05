Evolution
=========

This page documents modusa's journey and how it progressed over time. I will be as open as possible discussing how I felt building modusa, the decisions I made over time. This is to help all those people out there who might be at a different phase of building something meaningful.

.. note::

	Although I use LLMs to document stuff, correct grammatical errors but this page will be free of any sort of such activity. I want it to be just me and you.


April 21, 2025: Left Job
------------------------
I am leaving my full time job as a research assistant at a Digital Audio Lab, Mumbai, India after working for 1 year with them. I am thinking of working as a freelancer than dedicating full time. I am doing this so that I can look into other avenues and figure out how can I contribute better to the community.

Jun 5, 2025: Inception
----------------------
I had been working on audio domain (particularly music) for quite some years and had some pain points that I always wished somebody would have solved. I am thinking of solving it myself than waiting for someone else.

.. admonition:: **Pain points leading to 'modusa':**

	- We use numpy for numerical operations that are completely mathematical and index based (does not contain any semantics). The axis does not mean anything and every operation you do that changes the shape of the array, you need to recompute the new axis. This needs to be automated if you are dealing with physical signals that has meaning.
	- We use matplotlib for plotting which again is a general purpose library for visualisation. It will be much more user friendly if every physical signal automatically knows how to plot itself.
	- Same goes for scipy for signal based operations, all these libraries are built to provide us a general framework.
	- 'modusa' library is trying to build on top of these libraries but with more meaningful semantic-rich operations for physical signals that you are working on and providing you the right tools for any physical signal based on the signal space it is currently in.
	- Another problem working with non-semantic libraries are that you can end up performing operations between two signals from different signal space as long as they are compatible. 'modusa' puts check in place to avoid such mistakes.
	
.. admonition:: **Who should be the target users?**

	- **Researchers** working with digital signals.
	- **Educators** can use it for demonstrating signal processing concepts much easily without the hassle of bringing different tools from different places.
	- **Learners** can use it for experimenting and learning things hands-on.

Aug 1, 2025: Sad Realisation
----------------------------
After working for about 2 months straight where I did nothing else just literally waking up, coding, eating, coding again. Everyday I use to sleep with satisfaction that I built something and then the next morning just to wake up with the thought that whatever I did yesterday needs to handle these cases too. My brain was literally thinking about the problem even when I was asleep. I think that really started taking toll on my health.

Apart from all this, I think that I have gone way too far to make a general system for signals. The idea was to make it frictionless for people to use but I think it has drifted from the core mission and adding more friction rather than reducing it. The problem is that building any axis-aware system needs to be built from scratch piece by piece as every case is different, for example when you use numpy operations, the system should know what that each operation mean for the axis. This makes it much harder if you want to support all the numpy operations.

Although it feels like I lost all the work I put into this but there are still things that are useful, it is just that I moved in a direction to build tools that handle everything automatically. I am now changing the focus to building small utility tools that works together coherently with the existing libraries. I deleted almost the entire codebase and making it a library for utility tools that really helps people solving one very specific problem at a time. I have also decided to focus more on audio domain than trying to make something general.

So far, I have not yet shared my library with people as I wanted to share it once everything is properly set but I think that never happens. So I will be sharing it with them anyway to get feedbacks.

