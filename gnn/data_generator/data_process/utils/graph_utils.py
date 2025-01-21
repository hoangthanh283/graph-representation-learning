#!venv/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def check_intersect_range(x1, l1, x2, l2):
    if x1 > x2:
        x1, x2 = x2, x1
        l1, l2 = l2, l1
    return (x1 + l1) > x2


def get_intersect_range(x1, l1, x2, l2):
    if x1 > x2:
        x1, x2 = x2, x1
        l1, l2 = l2, l1
    if not check_intersect_range(x1, l1, x2, l2):
        return 0
    if (x1 + l1) > (x2 + l2):
        return l2
    else:
        return x1 + l1 - x2


def check_intersect_horizontal_proj(bbox1, bbox2):
    return check_intersect_range(bbox1[1], bbox1[3], bbox2[1], bbox2[3])


def check_intersect_vertical_proj(bbox1, bbox2):
    return check_intersect_range(bbox1[0], bbox1[2], bbox2[0], bbox2[2])


def get_intersect_range_horizontal_proj(bbox1, bbox2):
    return get_intersect_range(bbox1[1], bbox1[3], bbox2[1], bbox2[3])


def get_intersect_range_vertical_proj(bbox1, bbox2):
    return get_intersect_range(bbox1[0], bbox1[2], bbox2[0], bbox2[2])


class CellNode:

    threshhold_really_horizontal = 2.0
    threshold_really_vertical = 0.2

    def __init__(self, x, y, w, h, label=None, cell_type=None, is_sub=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_sub = is_sub
        self.parent = None
        self.drawed_rel = False
        self.label = label
        self.cell_type = cell_type

        # Get cell background

        self.list_texts = []
        self.sub_lines = []

        # Init adjacent
        self.lefts = []
        self.rights = []
        self.tops = []
        self.bottoms = []
        self.is_master_key = False
        self.is_value = False
        self.parent_key = []
        self.children_key = []
        self.drawed = False
        self.ocr_value = ""
        self.is_column_header = False
        self.column_header = None
        self.col = None
        self.row = None
        self.confidence = None

    def get_text(self):
        if self.is_sub:
            return "".join([str(text) for text in self.list_texts])
        else:
            return " ".join([subline.get_text() for subline in self.sub_lines])

    def get_aspect_ratio(self):
        return self.w / self.h

    def is_really_horizontal_cell(self):
        return self.get_aspect_ratio() > CellNode.threshhold_really_horizontal

    def is_really_vertical_cell(self):
        return self.get_aspect_ratio() < CellNode.threshold_really_vertical

    def get_bbox(self):
        return (self.x, self.y, self.w, self.h)

    def get_real_sub_lines(self):
        if len(self.sub_lines) == 0:
            if self.is_sub:
                return [self]
            else:
                return []
        else:
            return_list = []
            if self.is_sub:
                return_list.append(self)
            for child in self.sub_lines:
                return_list.extend(child.get_real_sub_lines())
            return return_list

    def is_left_of(self, other_cell, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights
        if other_cell in self.rights:
            return True

        if other_cell.x < self.x or not check_intersect_horizontal_proj(
            self.get_bbox(), other_cell.get_bbox()
        ):
            return False
        if get_intersect_range_horizontal_proj(
            self.get_bbox(), other_cell.get_bbox()
        ) > 0.9 * min(self.h, other_cell.h):
            if other_cell.x - self.x < 0.1 * min(self.w, other_cell.w):
                return True

        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [
            cell
            for cell in ref_cells
            if check_intersect_horizontal_proj(self.get_bbox(), cell.get_bbox())
            and (cell.x + cell.w) < other_cell.x + other_cell.w * 0.1
            and cell.x >= (self.x + self.w * 0.8)
            and check_intersect_horizontal_proj(
                self.get_bbox(), cell.get_bbox()
            )
        ]
        # 2. filters all the small overlapping cells
        ref_cells = [
            cell
            for cell in ref_cells
            if get_intersect_range_horizontal_proj(
                self.get_bbox(), cell.get_bbox()
            )
            > min(self.h, cell.h) / 5
        ]
        ref_cells = [
            cell
            for cell in ref_cells
            if get_intersect_range_horizontal_proj(
                cell.get_bbox(), other_cell.get_bbox()
            )
            > other_cell.h / 2
            or get_intersect_range_horizontal_proj(
                self.get_bbox(), cell.get_bbox()
            )
            > min(cell.h, self.h) * 0.8
        ]

        # 3. Check if there are any cells lies between this and other_cell
        if len(ref_cells) > 0:
            return False

        # 4. return results
        return True

    def is_right_of(self, other_cell, ref_cells):
        return other_cell.is_left_of(self, ref_cells)

    def is_top_of(self, other_cell, ref_cells):
        """Check if this cell is directly top of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights
        if other_cell in self.bottoms:
            return True

        if other_cell.y < self.y or not check_intersect_vertical_proj(
            self.get_bbox(), other_cell.get_bbox()
        ):
            return False

        if (
            get_intersect_range_vertical_proj(
                self.get_bbox(), other_cell.get_bbox()
            )
            < min(self.w, other_cell.w) / 5
        ):
            return False
        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [
            cell
            for cell in ref_cells
            if check_intersect_vertical_proj(self.get_bbox(), cell.get_bbox())
            and (cell.y + cell.h) < other_cell.y + other_cell.h * 0.1
            and cell.y >= (self.y + self.h * 0.8)
            and check_intersect_vertical_proj(self.get_bbox(), cell.get_bbox())
        ]
        # 2. filters all the small overlapping cells
        ref_cells = [
            cell
            for cell in ref_cells
            if get_intersect_range_vertical_proj(
                self.get_bbox(), cell.get_bbox()
            )
            > min(self.w, cell.w) / 5
        ]
        ref_cells = [
            cell
            for cell in ref_cells
            if get_intersect_range_vertical_proj(
                cell.get_bbox(), other_cell.get_bbox()
            )
            > other_cell.w / 2
            or get_intersect_range_vertical_proj(
                self.get_bbox(), cell.get_bbox()
            )
            > min(self.w, cell.w) * 0.8
        ]

        # 3. Check if there are any cells lies between this and other_cell
        if len(ref_cells) > 0:
            return False

        # 4. return result
        return True

    def set_text(self, text):
        self.ocr_value = text
        self.list_texts = [text]

    def __getitem__(self, key):
        return self.get_bbox()[key]


def get_cell_from_cell_list(cell_list_dict):
    """ Convert a list of cell info dictionary to list CellNode object

    Args:
        cell_list_dict (list): List of cell info in dictionary form.
        img

    """
    cell_list = []
    list_sub_cell = []
    for index, cell_data in enumerate(cell_list_dict):
        # Initialize new cell

        cell_bbox = cell_data["location"]
        cell_label = cell_data.get("label")
        cell_type = cell_data.get("type")

        xs = [p[0] for p in cell_bbox]
        ys = [p[1] for p in cell_bbox]

        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width = right - left
        height = bottom - top

        new_cell = CellNode(
            left, top, width + 1, height + 1, cell_label, cell_type,
        )
        new_cell.ocr_value = cell_data.get("text", "")
        new_cell.list_texts = [cell_data.get("text", "")]
        new_cell.confidence = cell_data.get("confidence", "")

        if cell_type not in ["cell", "table"]:  # is textline
            list_sub_cell.append(new_cell)
            list_sub_cell[-1].name = f"text_line{index}"
            list_sub_cell[-1].is_sub = True
        elif cell_type == "cell":
            cell_list.append(new_cell)
            cell_list[-1].name = f"cell_{index}"

    final_lines_with_no_parents = []
    while len(list_sub_cell) > 0:
        is_collied = False
        for cell in cell_list:
            if len(list_sub_cell) == 0:
                break
            if (
                cell.name.split("_")[0] == list_sub_cell[0].name.split("_")[0]
                and cell.name.split("_")[1]
                == list_sub_cell[0].name.split("_")[1]
            ):
                cell.sub_lines.append(list_sub_cell.pop())
                cell.sub_lines[-1].parent = cell
                is_collied = True
        if len(list_sub_cell) == 0:
            break
        if not is_collied:
            final_lines_with_no_parents.append(list_sub_cell.pop(0))
    cell_list.extend(final_lines_with_no_parents)
    return cell_list


def _get_v_intersec(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = max(0, min(y1_2 - y2_1, y2_2 - y1_1))
    return ret


def _get_v_union(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = min(h1 + h2, max(y2_2 - y1_1, y1_2 - y2_1))
    return ret


def _get_h_intersec(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    x1_2 = x1_1 + w1
    x2_2 = x2_1 + w2
    ret = max(0, min(x1_2 - x2_1, x2_2 - x1_1))
    return ret


def _get_h_union(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    x1_2 = x1_1 + w1
    x2_2 = x2_1 + w2
    ret = min(w1 + w2, max(x2_2 - x1_1, x1_2 - x2_1))
    return ret


def get_nearest_line(cr_line, list_lines, dr="l", thresh=50000):
    line_loc = cr_line.get_bbox()
    ret = None
    dt = thresh
    for line in list_lines:
        text = line.get_text()
        if text == "":
            continue
        loc = line.get_bbox()
        if dr == "r" or dr == "l":
            if _get_v_intersec(loc, line_loc) <= 0.3 * _get_v_union(
                loc, line_loc
            ):
                continue
        elif dr == "t" or dr == "b":
            d = min(
                abs(loc[1] - line_loc[1] - line_loc[3]),
                abs(line_loc[1] - loc[1] - loc[3]),
            )

            # 0.1 * _get_h_union(loc, line_loc):
            if _get_h_intersec(loc, line_loc) <= 0:
                if dr == "t" and line_loc[1] > loc[1]:
                    continue
                if not (
                    dr == "t"
                    and d < 0.5 * line_loc[3]
                    and line_loc[1] + 1.3 * line_loc[3] > loc[1]
                ):
                    continue

        dist = dt + 1
        if dr == "r":
            if loc[0] > line_loc[0]:
                dist = loc[0] - line_loc[0] - line_loc[2]
        elif dr == "l":
            if loc[0] < line_loc[0]:
                dist = line_loc[0] - loc[0] - loc[2]
        elif dr == "b":
            if loc[1] > line_loc[1]:
                dist = loc[1] - line_loc[1] - line_loc[3]
        elif dr == "t":
            if loc[1] < line_loc[1]:
                dist = line_loc[1] - loc[1] - loc[3]
        if dist < dt:
            ret = line
            dt = dist
    return ret


class Edge:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


class Column:
    def __init__(self, cell_list):
        self.cell_list = cell_list
        self.x = min([c.x for c in cell_list])
        self.y = min([c.y for c in cell_list])
        self.w = cell_list[0].w
        self.h = sum([c.h for c in cell_list])


class Row:
    def __init__(self, cell_list):
        self.cell_list = cell_list
        self.x = min([c.x for c in cell_list])
        self.y = min([c.y for c in cell_list])
        self.h = cell_list[0].h
        self.w = sum([c.w for c in cell_list])


class Graph:
    """ There are 6 heuristic relations in the adjacency matrix:
        left-right
        right-left
        top-bottom
        bottom-top
        child
        parrent
    """
    edge_labels = ["lr", "rl", "tb", "bt", "child", "parent"]

    def __init__(self, kv_input, edge_type="normal_binary"):
        self.org_items = get_cell_from_cell_list(kv_input)

        self.table_cells = [item for item in self.org_items if not item.is_sub]
        self.text_lines = []
        for item in self.org_items:
            self.text_lines.extend(item.get_real_sub_lines())

        self._detect_row()
        self._detect_column()

        self.nodes = self.text_lines + self.table_cells + self.rows + self.cols

        self.edges = []
        self.build_edges()
        self._get_adj_matrix(edge_type)

    def build_edges(self):
        for cell_list in (self.text_lines, self.table_cells):
            # 1. left-right
            cell_list_top_down = sorted(cell_list, key=lambda cell: cell.y)
            cell_list_left_right = sorted(cell_list, key=lambda cell: cell.x)
            # 1.1 Check this cell with every cell to the right of it
            # TODO: More effective iteration algo e.g: cached collisions matrix
            self._build_left_right_edges(cell_list_top_down)
            # 2. top-down
            self._build_top_down_edges(cell_list_left_right)
        self._build_child_parent_edges()

        # clean left-right edges
        self._clean_left_right_edges()
        # clean top-bot edges
        self._clean_top_bot_edges()

    def _build_left_right_edges(self, cell_list_top_down):
        for cell in cell_list_top_down:
            cell_collide = [
                other_cell
                for other_cell in cell_list_top_down
                if other_cell.x >= cell.x
                and check_intersect_horizontal_proj(
                    cell.get_bbox(), other_cell.get_bbox()
                )
                and cell != other_cell
            ]
            cell_collide = [
                other_cell
                for other_cell in cell_collide
                if get_intersect_range_horizontal_proj(
                    cell.get_bbox(), other_cell.get_bbox()
                )
                > min(cell.h, other_cell.h) * 0.4
            ]

            for other_cell in cell_collide:
                if (
                    cell.is_left_of(other_cell, cell_collide)
                    and other_cell not in cell.rights
                ):
                    self.edges.append(
                        Edge(cell, other_cell, self.edge_labels.index("lr"))
                    )
                    self.edges.append(
                        Edge(other_cell, cell, self.edge_labels.index("rl"))
                    )
                    cell.rights.append(other_cell)
                    other_cell.lefts.append(cell)

    def _clean_left_right_edges(self):
        for cell in self.text_lines:
            if len(cell.lefts) <= 1:
                continue
            left_cells = sorted(cell.lefts, key=lambda x: x.x)
            removes = [
                c
                for c in left_cells
                if c.x + c.w > cell.x and c.x > cell.x - 0.5 * cell.h
            ]
            left_cells = [c for c in left_cells if c not in removes]
            # cluster these cell into column:

            columns = []
            column_cells = []
            # column_x = left_cells[0].x
            for c in left_cells:
                its = 0
                union = 100
                if column_cells:
                    its = get_intersect_range_vertical_proj(
                        column_cells[-1].get_bbox(), c.get_bbox()
                    )
                    union = min(column_cells[-1].w, c.w)
                if its > 0.5 * union:
                    column_cells.append(c)
                    continue
                else:
                    if len(column_cells) > 0:
                        columns.append(column_cells)
                    column_cells = [c]
                    # column_x = c.x
            if column_cells:
                columns.append(column_cells)

            # left_cells to keep:
            if len(columns) > 0:
                real_lefts = columns[-1]
            else:
                real_lefts = []
            removes += [c for c in left_cells if c not in real_lefts]
            remove_edges = []
            for c in removes:
                c.rights.remove(cell)
                for e in self.edges:
                    if (
                        e.start == c
                        and e.end == cell
                        and e.label == self.edge_labels.index("lr")
                    ):
                        remove_edges.append(e)
                    if (
                        e.start == cell
                        and e.end == c
                        and e.label == self.edge_labels.index("rl")
                    ):
                        remove_edges.append(e)
            [self.edges.remove(e) for e in remove_edges]

            cell.lefts = real_lefts

    def _build_top_down_edges_1(self, cell_list_left_right):
        """Legacy build top down edeges."""
        for cell in cell_list_left_right:
            cell_collide = [
                other_cell
                for other_cell in cell_list_left_right
                if other_cell.y > cell.y + cell.h * 0.6
                and check_intersect_vertical_proj(
                    cell.get_bbox(), other_cell.get_bbox()
                )
                and cell != other_cell
            ]
            for other_cell in cell_collide:
                if (
                    cell.is_top_of(other_cell, cell_collide)
                    and other_cell not in cell.bottoms
                ):
                    self.edges.append(
                        Edge(cell, other_cell, self.edge_labels.index("tb"))
                    )
                    self.edges.append(
                        Edge(other_cell, cell, self.edge_labels.index("bt"))
                    )
                    cell.bottoms.append(other_cell)
                    other_cell.tops.append(cell)

    def _build_top_down_edges(self, cell_list_left_right):
        for cell in cell_list_left_right:
            top_cell = get_nearest_line(cell, cell_list_left_right, "t")
            if top_cell:
                self.edges.append(
                    Edge(top_cell, cell, self.edge_labels.index("tb"))
                )
                self.edges.append(
                    Edge(cell, top_cell, self.edge_labels.index("bt"))
                )
                cell.tops.append(top_cell)
                top_cell.bottoms.append(cell)

    def _clean_top_bot_edges(self):
        for cell in self.text_lines:
            if len(cell.tops) <= 1:
                continue
            top_cells = sorted(cell.tops, key=lambda x: x.y)

            rows = []
            row_cells = []
            for c in top_cells:
                its = 0
                union = 10000
                if row_cells:
                    its = get_intersect_range_horizontal_proj(
                        row_cells[-1].get_bbox(), c.get_bbox()
                    )
                    union = min(row_cells[-1].w, c.w)
                if its > 0.5 * union:
                    row_cells.append(c)
                    continue
                else:
                    if len(row_cells) > 0:
                        rows.append(row_cells)
                    row_cells = [c]
            if row_cells:
                rows.append(row_cells)

            # left_cells to keep:
            real_tops = rows[-1]
            removes = [c for c in top_cells if c not in real_tops]
            remove_edges = []
            for c in removes:
                c.bottoms.remove(cell)
                for e in self.edges:
                    if (
                        e.start == c
                        and e.end == cell
                        and e.label == self.edge_labels.index("tb")
                    ):
                        remove_edges.append(e)
                    if (
                        e.start == cell
                        and e.end == c
                        and e.label == self.edge_labels.index("bt")
                    ):
                        remove_edges.append(e)
            [self.edges.remove(e) for e in remove_edges]

            cell.tops = real_tops

    def _build_child_parent_edges(self):
        for cell in self.text_lines:
            parent = cell.parent
            if not parent:
                continue
            childs = parent.sub_lines
            for ch in childs:
                self.edges.append(
                    Edge(ch, parent, self.edge_labels.index("parent"))
                )
                self.edges.append(
                    Edge(parent, ch, self.edge_labels.index("child"))
                )

        for row in self.rows:
            for ch in row.cell_list:
                self.edges.append(
                    Edge(ch, row, self.edge_labels.index("parent"))
                )
                self.edges.append(
                    Edge(row, ch, self.edge_labels.index("child"))
                )

        for col in self.cols:
            for ch in col.cell_list:
                self.edges.append(
                    Edge(ch, col, self.edge_labels.index("parent"))
                )
                self.edges.append(
                    Edge(col, ch, self.edge_labels.index("child"))
                )

    def _detect_column(self):
        self.cols = []
        used_cells = []
        for cell in self.table_cells:
            aligns = []
            if cell in used_cells:
                continue
            aligns.append(cell)
            p_margin = cell.w / 4
            w_margin = cell.w / 6

            for o_cell in self.table_cells:
                if o_cell in used_cells:
                    continue
                if o_cell == cell:
                    continue

                if (
                    abs(o_cell.x - cell.x) <= p_margin
                    and abs(o_cell.w - cell.w) <= w_margin
                ):
                    aligns.append(o_cell)
            used_cells.extend(aligns)
            if len(aligns) > 1:
                new_col = Column(aligns)
                for c in aligns:
                    c.col = new_col
                self.cols.append(new_col)

    def _detect_row(self):
        self.rows = []
        used_cells = []
        for cell in self.table_cells:
            aligns = []
            if cell in used_cells:
                continue
            aligns.append(cell)
            p_margin = cell.h / 2
            h_margin = cell.h / 4

            for o_cell in self.table_cells:
                if o_cell in used_cells:
                    continue
                if o_cell == cell:
                    continue

                if (
                    abs(o_cell.y - cell.y) <= p_margin
                    and abs(o_cell.h - cell.h) <= h_margin
                ):
                    aligns.append(o_cell)
            used_cells.extend(aligns)
            if len(aligns) > 1:
                new_row = Row(aligns)
                for c in aligns:
                    c.row = new_row
                self.rows.append(new_row)

    def _get_adj_matrix(self, edge_type="normal_binary"):
        def scale_coor(node):
            scale_x1 = (node.x - min_x) / max_delta_x
            scale_y1 = (node.y - min_y) / max_delta_y
            scale_x1b = (node.x + node.w - min_x) / max_delta_x
            scale_y1b = (node.y + node.h - min_y) / max_delta_y
            return scale_x1, scale_y1, scale_x1b, scale_y1b

        def dist(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        def rect_distance(rect1, rect2):
            x1, y1, x1b, y1b = rect1
            x2, y2, x2b, y2b = rect2

            left = x2b < x1
            right = x1b < x2
            bottom = y2b < y1
            top = y1b < y2

            if top and left:
                return dist(x1, y1b, x2b, y2)
            elif left and bottom:
                return dist(x1, y1, x2b, y2b)
            elif bottom and right:
                return dist(x1b, y1, x2, y2b)
            elif right and top:
                return dist(x1b, y1b, x2, y2)
            elif left:
                return x1 - x2b
            elif right:
                return x2 - x1b
            elif bottom:
                return y1 - y2b
            elif top:
                return y2 - y1b

            return 0.0

        adj = np.zeros(
            (len(self.nodes), len(self.edge_labels), len(self.nodes))
        )

        max_x = np.max([n.x + n.w for n in self.nodes])
        max_y = np.max([n.y + n.h for n in self.nodes])
        min_x = np.min([n.x for n in self.nodes])
        min_y = np.min([n.y for n in self.nodes])

        max_delta_x = np.abs(max_x - min_x)
        max_delta_y = np.abs(max_y - min_y)

        if "fc_similarity" == edge_type:
            # Fully connected
            for i in range(len(self.nodes)):
                for j in range(i, len(self.nodes)):
                    if i == j:
                        adj[i, :, j] = 1
                    else:
                        rect1 = scale_coor(self.nodes[i])
                        rect2 = scale_coor(self.nodes[j])
                        edist = np.abs(rect_distance(rect1, rect2))
                        adj[i, :, j] = (1 - (edist / np.sqrt(2))) ** 2
                        adj[j, :, i] = (1 - (edist / np.sqrt(2))) ** 2

            # ---------------------------------
        elif "fc_binary" == edge_type:
            # Fully connected
            for i in range(len(self.nodes)):
                for j in range(i, len(self.nodes)):
                    if i == j:
                        adj[i, :, j] = 1
                    else:
                        rect1 = scale_coor(self.nodes[i])
                        rect2 = scale_coor(self.nodes[j])
                        edist = np.abs(rect_distance(rect1, rect2))
                        adj[i, :, j] = 1
                        adj[j, :, i] = 1
            # ---------------------------------
        elif "normal_binary" == edge_type:
            for edge in self.edges:
                # Distance Euclidean distance based similarity is bounded in range [0, 1]
                # where 0 mean the two objects are to far from each other, and 1 means
                # they have the same central gravity.
                start = self.nodes.index(edge.start)
                end = self.nodes.index(edge.end)

                adj[start, edge.label, end] = 1

        else:
            raise Exception("Invalid edge type: " + str(edge_type))

        self.adj = adj.astype(np.float16)
