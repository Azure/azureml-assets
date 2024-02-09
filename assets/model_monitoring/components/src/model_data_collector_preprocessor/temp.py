import bisect

class Span:
    def __init__(self, span_id, parent_id, start, end):
        self.span_id = span_id
        self.parent_id = parent_id
        self.start = start
        self.end = end
        self.children = []

    def show(self, indent=0):
        print(f"{' '*indent}[{self.span_id}({self.start}, {self.end})]")
        for c in self.children:
            c.show(indent+4)

    def __iter__(self):
        for child_span in self.children:
            for span in child_span:
                yield span
        yield self

class SpanTree:
    def __init__(self, spans):
        self.root_span = self._construct_span_tree(spans)

    def _construct_span_tree(sel, spans):
        # construct a dict with span_id as key and span as value
        span_map = {}
        for span in spans:
            span_map[span.span_id] = span
        
        for span in span_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = span_map.get(parent_id)
                # insert in order of end
                bisect.insort(parent_span.children, span, key=lambda s: s.end)
        return root_span
    
    def show(self):
        if self.root_span is None:
            return
        self.root_span.show()
    
    def __iter__(self):
        for span in self.root_span.__iter__():
            yield span


# testing
s0 = Span("0", None, 0, 100)
s00 = Span("00", "0", 5, 30)
s01 = Span("01", "0", 35, 60)
s02 = Span("02", "0", 65, 90)
s010 = Span("010", "01", 40, 50)
spans = [s0, s02, s010, s00, s01]

# construct tree
span_tree = SpanTree(spans)
span_tree.show()
print()
# iterate over span tree, per end order
for span in span_tree:
    print(f"{span.span_id}:\t({span.start},\t{span.end})")